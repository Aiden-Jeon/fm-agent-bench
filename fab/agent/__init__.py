"""MLE-bench 트랙 B: 하네스 × LLM 에이전트를 Databricks 서버리스 GPU 에서 실행·회수·채점.

    uv run fab mleb agent run spaceship-titanic claude-code-sonnet          # 제출 (기다리지 않음)
    uv run fab mleb agent run nomad2018-predict-transparent-conductors aide-gpt --rep 2
    uv run fab mleb agent run spaceship-titanic claude-code-sonnet --continue  # 별도 설정 +cont (kaggle-master#35)
    uv run fab mleb agent run nomad2018-predict-transparent-conductors codex-gpt --fm  # C2: TFM 을 깐 환경 (+fm)
    uv run fab mleb agent list                                              # 제출한 실행과 처리 상태 (Volume 의 state/agent/)
    uv run fab sync                                                         # 채점된 결과 → results/mlebench_runs.csv

실행은 제출한 뒤 맥북을 꺼도 원격에서 계속 돎. 끝난 뒤 `fab remote kick` 으로 정리 작업(fab/remote/reconcile.py)이 회수·채점·사후 검사.

실행 환경 (docs/plans/mle-bench.md, 2026-11-09 결정):
  - 작업은 전용 SP 신원으로 돎 (run_as). SP 권한은 scratch Volume 읽기·쓰기 + (모든 사용자처럼) FM API 조회.
    FM API 시스템 엔드포인트에는 ACL 을 걸 수 없어서, 실행기 프록시가 모델을 설정값으로 고정하고 요청을 기록함
  - 정답(private)은 로컬 mlebench 캐시와 채점용 내 Volume(databricks.local.yaml 의 volume)에만 있음
    → SP 의 scratch Volume 에는 public 만 올림. 채점은 원격 정리 작업이 내 신원으로 (fab/bench/mlebench/remote.py)
  - A10 1장, 벽시계 hours 시간. 원격 쪽은 fab/agent/runner.py
  - 기본은 공식 설정: 하네스가 스스로 끝나면 그대로 끝. --continue 는 별도 설정(<config>+cont)으로, 시간이 남았는데 끝나면
    같은 세션에 "남은 시간 N분, 계속 개선하라" 를 보내 이어 감 (세 하네스 같은 규칙, 2026-11-10 결정, kaggle-master#35)
  - --fm 은 C2 (<config>+fm, docs/plans/roadmap.md 의 P5): 환경에 TFM(tabpfn_v2·tabicl·kumo)을 깔고 가중치를 받아 둔 뒤
    지시문에 이름과 사용 예(fm_notes.txt)를 더함. 나머지는 공식 설정과 같음. 준비는 하네스 시작 전 (runner.py 의 'C2')

databricks.local.yaml 의 mleb_agent 블록 (git 제외):
    mleb_agent:
      run_as: <SP application id>
      volume: /Volumes/<catalog>/<schema>/<volume>     # SP 가 읽고 쓰는 scratch Volume
      experiment: <MLflow experiment 이름>             # 선택. 기본 fab-mleb-agent
      traces_table: <catalog>.<schema>.<prefix>_otel_spans  # 선택. LLM 호출·하네스 trace 를 이 UC 표로 (README 'trace')
      gateway: true          # 선택. LLM 요청을 FM API 대신 Unity Gateway 로 → system.ai_gateway.usage 에 남음
  gateway 는 실행 조건(분당 한도·경로)이 바뀔 수 있어 run.json·runs.json·결과 표(llm_route)에 남김. 자세한 건 runner.py 의
  'trace'. traces_table 은 기록만 더함 (요청 경로·내용은 그대로)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fab.agent.fm_audit import HEARTBEAT, WATCH
from fab.core.results import OUTPUTS, RESULTS, table_lock
from fab.remote import items, submit_item
from fab.remote import settings as remote_settings

# 설정 이름 → (하네스, FM API 엔드포인트, 하네스에 넘길 모델 이름). AIDE 는 gpt- 로 시작해야 OpenAI 백엔드를 써서 별칭
# (별칭 gpt-km-* 는 kaggle-master 때 이름 그대로: 기록된 실행의 proxy.jsonl 과 맞춰야 사후 검사(other_models)가 같음)
SONNET, GPT = "databricks-claude-sonnet-5-5", "databricks-gpt-6-1-terra"
CONFIGS = {
    "aide-sonnet": ("aide", SONNET, "gpt-km-sonnet"),
    "aide-gpt": ("aide", GPT, "gpt-km-gpt"),
    "claude-code-sonnet": ("claude-code", SONNET, SONNET),
    "codex-gpt": ("codex", GPT, GPT),
}
CONTINUE_SUFFIX = "+cont"  # --continue 로 돌린 별도 설정의 이름 꼬리표 (공식 설정과 결과 표에서 섞이지 않게)
FM_SUFFIX = "+fm"  # --fm (C2): 환경에 TFM 을 깐 설정
# --fm 의 작업 시간 여유 더하기 (하네스 1시간 밖): TFM pip 설치 + apt (runner.py 의 APT_TIMEOUT × 2) + 사전 점검
# (FM_WARMUP_TIMEOUT × 3) 최악 합 약 60분에 여유. 작업 시간 상한이라 넉넉해도 비용은 실제로 쓴 만큼
FM_EXTRA_MINUTES = 75
GATEWAY_SUFFIX = "+gw"  # gateway: true 로 돌린 실행의 꼬리표 (분당 한도·거부 조건이 FM API 와 달라 섞지 않게, llm_route 열과 같은 뜻)
HARNESS_VERSIONS = {"aide": "v6.3.3", "claude-code": "2.1.319", "codex": "0.167.0"}  # MLE-bench 의 AIDE 포크 태그
ACCELERATOR = "GPU_1xA10"
AGENT_RUNS = OUTPUTS / "_mlebench" / "agents"  # 회수한 실행: <key>/ (원격 정리 작업에서는 내 Volume 의 outputs/ 아래)
TABLE = RESULTS / "mlebench_runs.csv"  # 에이전트 실행별 채점 (git 추적)
# TabArena 에이전트 실행별 채점 (git 추적, 열은 mlebench_runs 와 같고 대회 열만 다름). 위치는 fab.bench.tabarena 에서 정함
from fab.bench.tabarena import RUNS as TA_TABLE  # noqa: E402
HERE = Path(__file__).parent
# 사후 검사: 에이전트가 TFM 을 썼는지, Kaggle 풀이·토론을 봤는지 (기록에서 찾음)
# 단어 경계: "carte" 가 "Cartesian" 에 걸렸음 (2026-11-09 nomad Codex). Kumo 는 코드에 KumoTabular 로 나옴 → kumo 로 셈
TFM_PATTERN = re.compile(r"\b(tabpfn|tabicl|kumo(?:tabular)?|autogluon|mitra|limix|exaone|tabdpt|carte)\b", re.I)
# 코드의 import 문 (주석의 이름은 안 셈): `import a, b as c` 는 모듈마다, `from a.b import c` 는 a.
# 모듈 → TFM 이름은 실행 중 기록과 같은 표 (fm_audit.WATCH, Kumo 의 모듈은 sdm)
IMPORT_LINE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\b|import\s+([\w., ]+))", re.M)
# databricks fs ls 가 경로 없음으로 실패할 때의 메시지 (그 밖의 실패는 일시적 오류로 봄)
MISSING = re.compile(r"no such (?:file|directory)|not found|does not exist|RESOURCE_DOES_NOT_EXIST", re.I)
KAGGLE_PATTERN = re.compile(r"kaggle\.com/(?:code|competitions/[^/\s\"']+/(?:discussion|code))", re.I)
# 원본 데이터 접근 (TabArena 는 타깃까지 공개, docs/tabarena.md 5): OpenML·UCI·Kaggle 데이터셋 주소와 받는 API.
# from 은 import 문만 (2026-11-14 Codex 의 생각 글 "...trained on real datasets from OpenML" 이 걸렸음)
DATA_SOURCE_PATTERN = re.compile(
    r"openml\.org|\bimport\s+openml\b|\bfrom\s+openml(?:\.\w+)*\s+import\b|fetch_openml|archive\.ics\.uci\.edu|doi\.org/10\.24432|"
    r"\bucimlrepo\b|kaggle\.com/(?:datasets|c/|competitions/[^/\s\"']+/data)|\bkagglehub\b|kaggle\s+(?:datasets|competitions)\s+download",
    re.I)
# 실행한 명령 안의 pip 설치 (pip_target). 하네스 기록의 명령 칸 (Codex·Claude Code 의 command) 과 작업 디렉터리 코드의 줄에서 찾음.
# 생각·대화 글과 명령 출력 (aggregated_output: README 의 "pip install tabpfn" 등) 은 실행이 아니라서 안 봄
PIP_INSTALL = re.compile(r"\bpip3?\s+install\b|-m\s+pip\s+install\b|\buv\s+pip\s+install\b")
COMMAND_KEYS = {"command", "cmd"}
# 값을 하나 받는 pip install 옵션 (값을 패키지 이름으로 세지 않게)
PIP_VALUE_OPTS = {"-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url", "--extra-index-url",
                  "-f", "--find-links", "-t", "--target", "--prefix", "--root", "--platform", "--python-version",
                  "--implementation", "--abi", "--only-binary", "--no-binary", "--progress-bar", "--upgrade-strategy",
                  "--src", "--cache-dir", "--log", "--timeout", "--retries", "--trusted-host", "--proxy", "-C",
                  "--config-settings", "--python", "-p", "--report", "--global-option", "--install-option"}
PIP_OUTSIDE_OPTS = {"-t", "--target", "--prefix", "--root"}  # 하네스 환경 밖 폴더에 까는 옵션 (pip freeze 에 안 나옴)
VENV_MAKE = re.compile(r"(?:-m\s+venv|\bvirtualenv|\buv\s+venv)\s+(?:-{1,2}[\w-]+(?:=\S+)?\s+)*([^\s;&|'\"]+)")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n[ \t]*\2[ \t]*(?=\n|$)", re.S)


def settings() -> dict:
    from fab.runner.databricks import LOCAL_CONFIG

    import yaml

    cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
    agent = cfg.get("mleb_agent") or {}
    if not (cfg.get("profile") and agent.get("run_as") and agent.get("volume")):
        raise SystemExit(f"{LOCAL_CONFIG.name} 에 profile 과 mleb_agent.run_as / mleb_agent.volume 필요 (fab/agent/__init__.py 설명)")
    return dict(profile=cfg["profile"], run_as=agent["run_as"], volume=agent["volume"].rstrip("/"),
                experiment=agent.get("experiment", "fab-mleb-agent"), gateway=bool(agent.get("gateway")),
                traces_table=agent.get("traces_table"))


def gateway_model(endpoint: str) -> str:
    """FM API 엔드포인트 이름 → Unity Gateway model service 이름 (databricks-claude-sonnet-5-5 → system.ai.claude-sonnet-5-5)."""
    return f"system.ai.{endpoint.removeprefix('databricks-')}"


def llm_route(info: dict) -> str:
    """실행의 LLM 경로. gateway 설정 이전 실행(runs.json 에 없음)은 FM API."""
    return "gateway" if info.get("gateway") else "fm_api"


def dbx(cfg, *args, capture=True, check=True, timeout=1200):
    res = subprocess.run(["databricks", *args, "-p", cfg["profile"]], text=True, capture_output=capture,
                         timeout=timeout)
    if check and res.returncode:  # capture 하면 CLI 의 에러 메시지가 묻히므로 같이 보여 줌
        raise SystemExit(f"databricks {' '.join(args[:2])} 실패 (exit {res.returncode}):\n{res.stderr or ''}")
    return res


def content_hash(folder: Path) -> str:
    """폴더 파일 (이름 · 내용, _ 로 시작하는 로컬 캐시 제외) 의 sha256 앞 12자리. tar 의 시각과 무관."""
    h = hashlib.sha256()
    for p in sorted(folder.iterdir()):
        if p.is_file() and not p.name.startswith("_"):
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def upload_data(cfg, cid: str, public: Path | None = None) -> str:
    """public 데이터를 tar 하나로 올림 (파일이 많은 대회도 한 번에). 이미 있으면 그대로.
    public: 올릴 폴더 (TabArena: 대회 data/, cid 는 익명 id). 이때 묶음 이름에 내용 해시를 붙여 (<cid>-<해시>.tgz) 설명·데이터를
    고친 뒤에는 새 묶음을 올림 (누출 검사를 한 로컬 파일 = 에이전트가 받는 묶음). 없으면 MLE-bench 의 public (이름은 <cid>.tgz)."""
    name = f"{cid}-{content_hash(public)}.tgz" if public is not None else f"{cid}.tgz"
    remote = f"{cfg['volume']}/data/{name}"
    if name in dbx(cfg, "fs", "ls", f"dbfs:{cfg['volume']}/data", check=False).stdout.split():
        return remote
    if public is None:
        from fab.bench.mlebench import _driver

        public = Path(_driver("paths", cid)["public"])
    if not public.is_dir() or not any(public.iterdir()):
        raise SystemExit(f"{public} 비어 있음 → 먼저 fab download (mlebench: {cid} 인 대회)")
    with tempfile.TemporaryDirectory() as d:
        tgz = Path(d) / name
        with tarfile.open(tgz, "w:gz") as tar:
            for p in sorted(public.iterdir()):
                if not p.name.startswith("_"):  # _features/ 등 로컬 캐시
                    tar.add(p, arcname=p.name)
        dbx(cfg, "fs", "mkdir", f"dbfs:{cfg['volume']}/data")
        dbx(cfg, "fs", "cp", "--overwrite", str(tgz), f"dbfs:{remote}", timeout=3600)
    print(f"데이터 업로드: {public} → {remote}")
    return remote


def read_jsonl(path: Path) -> list[dict]:
    """jsonl 의 줄들. 쓰는 도중 복사됐거나 작업이 죽어 잘린 줄은 건너뜀."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def check_continue(configs: list[str], cont: bool):
    """AIDE 는 자기 time_limit 까지 돌고 이어 갈 세션이 없음 → +cont 가 공식 설정과 같아서 거부.
    여러 설정을 한 번에 낼 때는 제출 전에 전부 확인 (앞 설정만 제출되고 중간에 멈추지 않게)."""
    if cont and (aide := [c for c in configs if CONFIGS[c][0] == "aide"]):
        raise SystemExit(f"--continue 는 AIDE 해당 없음 (일찍 끝나지 않음, 이어 갈 세션 없음): {', '.join(aide)}")


def ta_target(name: str):
    """TabArena 에이전트 대회 (r0f0, 설정에 anon_id) → (대회, 익명 id). 에이전트에게 보이는 곳에는 익명 id 만 씀."""
    from fab.core.competition import load_competition

    comp = load_competition(name)
    ta = comp.tabarena or {}
    if not ta.get("anon_id"):
        raise SystemExit(f"{name}: 설정에 tabarena.anon_id 가 없음 (에이전트는 r0f0 의 ta-<데이터셋> 대회만, 결정 2)")
    if ta.get("split") != "r0f0":
        raise SystemExit(f"{name}: 에이전트는 r0f0 만 ({ta.get('split')})")
    if not (comp.dir / "description.md").exists():
        raise SystemExit(f"{comp.dir}/description.md 없음")
    return comp, ta["anon_id"]


def leak_terms(comp) -> list[str]:
    """에이전트에게 보이면 안 되는 글: 데이터셋 이름 (원래·소문자·구분자 바꾼 것), 대회 이름, OpenML task·dataset id."""
    from fab.bench.tabarena import dataset_row

    ds = comp.tabarena["dataset"].lower()
    words = {ds, ds.replace("_", "-"), ds.replace("-", "_"), ds.replace("_", " "), ds.replace("-", " "), comp.name,
             str(comp.tabarena["task_id"]), str(dataset_row(comp.tabarena["dataset"])["dataset_id"])}
    return sorted(words)


def find_leaks(terms: list[str], texts: dict[str, str]) -> list[str]:
    """texts {어디: 글} 에서 terms 가 나오는 곳. 숫자 (OpenML id) 는 앞뒤가 숫자가 아닐 때만 (긴 숫자의 일부는 안 셈),
    이름은 앞뒤가 영문·숫자가 아닐 때만 (대소문자 무시. 'mic' 가 'dynamic' 에, 'houses' 가 'warehouses' 에 걸리지 않게)."""
    found = []
    for where, text in texts.items():
        for t in terms:
            edge = r"\d" if t.isdigit() else r"[a-z0-9]"
            if re.search(rf"(?<!{edge}){re.escape(t)}(?!{edge})", text, re.I):
                found.append(f"{where}: '{t}'")
    return found


def leak_check(comp, files: list[Path], texts: dict[str, str]):
    """에이전트 노드에 가는 것 (코드 묶음 파일·run.json·데이터 파일 이름·설명) 에 데이터셋을 알려 주는 글이 있으면 멈춤.
    데이터 파일은 이름과 description.md 만 봄 (피처 이름은 그대로 주기로 함, 결정 3)."""
    texts = dict(texts, **{str(f): f.read_text(errors="replace") for f in files})
    if found := find_leaks(leak_terms(comp), texts):
        raise SystemExit("에이전트에게 보이는 곳에 데이터셋을 알려 주는 글이 있음 → 제출 안 함:\n  " + "\n  ".join(found))


def submit(cid: str, config: str, rep: int, hours: float, cont: bool = False, fm: bool = False, comp=None) -> str:
    """comp: TabArena 대회 (cid = 익명 id). 데이터는 그 대회의 data/ (public), 대회 이름은 내 Volume 의 제출 기록에만."""
    from fab.bench.mlebench import MLEBENCH_DIR

    harness, model, alias = CONFIGS[config]
    check_continue([config], cont)
    cfg, main_cfg = settings(), remote_settings()
    label = (f"{config}{CONTINUE_SUFFIX if cont else ''}{FM_SUFFIX if fm else ''}"
             f"{GATEWAY_SUFFIX if cfg['gateway'] else ''}")
    if comp is not None:  # 올리기 전에 데이터 파일 이름·설명 확인
        leak_check(comp, [comp.data_dir / "description.md"],
                   {"data files": " ".join(p.name for p in comp.data_dir.iterdir()), "cid": cid})
    data = upload_data(cfg, cid, comp.data_dir if comp is not None else None)
    # 키는 Volume 경로·command_path 에 들어감 → '+' 불가 (jobs submit 이 거부, 2026-11-10). 설정 이름(label)은 +cont 그대로
    key = f"{cid}__{label.replace('+', '-')}__r{rep}__{datetime.now():%Y%m%d-%H%M%S}"
    out = f"{cfg['volume']}/runs/{key}"
    run = dict(key=key, cid=cid, harness=harness, model=model, alias=alias, hours=hours, data_tgz=data, out=out,
               harness_versions=HARNESS_VERSIONS, **{"continue": cont}, fm=fm, gateway=cfg["gateway"],
               upstream_model=gateway_model(model) if cfg["gateway"] else model, traces_table=cfg["traces_table"])
    with tempfile.TemporaryDirectory() as d:
        code = Path(d) / "code"
        code.mkdir()
        shutil.copy(HERE / "runner.py", code)
        for name in ("additional_notes.txt", "fm_audit.py", "fm_notes.txt", "fm_warmup.py"):
            shutil.copy(HERE / name, code)
        shutil.copy(MLEBENCH_DIR / "environment" / "instructions.txt", code)  # 공식 지시문 그대로 (경로는 실행기가 바꿈)
        shutil.copy(MLEBENCH_DIR / "environment" / "validate_submission.sh", code)
        (code / "run.json").write_text(json.dumps(run, indent=2))
        if comp is not None:  # 에이전트 노드에 가는 코드 묶음 전부 (대응표가 섞이지 않았는지)
            leak_check(comp, sorted(p for p in code.iterdir() if p.is_file()), {"run_name": f"fab-mleb-{key}"})
        # 실행마다 code/<key>/ 에 code.tgz·command.sh·env_vars.json·secret_env_vars.json. AI Runtime 런처는
        # Volume 의 command_path 옆 두 json 을 꼭 읽음 (없으면 ENV_VARS_FILE_READ_FAILED). air CLI 를 안 거치니 직접 둠
        launch = Path(d) / "launch"
        launch.mkdir()
        tgz, cmd = launch / "code.tgz", launch / "command.sh"
        with tarfile.open(tgz, "w:gz") as tar:  # 최상위 폴더 하나로 묶음: 런처는 첫 항목을 $CODE_SOURCE_PATH 로 잡음
            tar.add(code, arcname="code")
        cmd.write_text("python $CODE_SOURCE_PATH/runner.py\n")
        for name in ("env_vars.json", "secret_env_vars.json"):  # 둘 다 빈 배열 (객체면 ENV_VARS_PARSE_FAILED). 비밀값은 안 넘김
            (launch / name).write_text("[]\n")
        remote = f"{cfg['volume']}/code/{key}"
        dbx(cfg, "fs", "mkdir", f"dbfs:{remote}")
        for f in sorted(launch.iterdir()):
            dbx(cfg, "fs", "cp", "--overwrite", str(f), f"dbfs:{remote}/{f.name}")
        job = {
            "run_name": f"fab-mleb-{key}"[:100],
            "run_as": {"service_principal_name": cfg["run_as"]},
            "timeout_seconds": int(hours * 3600) + (45 + (FM_EXTRA_MINUTES if fm else 0)) * 60,  # 하네스 + 설치·회수 여유
            "tasks": [{"task_key": "agent", "environment_key": "default", "max_retries": 0,
                       "ai_runtime_task": {
                           "experiment": cfg["experiment"], "mlflow_run": key[:100],
                           "deployments": [{"command_path": f"{remote}/{cmd.name}",
                                            "compute": {"accelerator_type": ACCELERATOR, "accelerator_count": 1}}],
                           "code_source_path": f"{remote}/{tgz.name}"}}],
            "environments": [{"environment_key": "default", "spec": {"environment_version": "6"}}],
        }
        (Path(d) / "job.json").write_text(json.dumps(job))
        res = dbx(cfg, "jobs", "submit", "--no-wait", "--json", f"@{Path(d) / 'job.json'}", "-o", "json")
    run_id = json.loads(res.stdout[res.stdout.index("{"):])["run_id"]
    extra = dict(bench="tabarena", comp=comp.name) if comp is not None else {}  # 내 Volume 의 제출 기록에만
    submit_item(main_cfg, "agent", key, dict(run, run_id=run_id, config=label, rep=rep, **extra))
    print(f"제출: {key} (run_id {run_id})" + (f" · LLM 경로 Unity Gateway ({run['upstream_model']})" if cfg["gateway"] else "")
          + (f" · trace → {cfg['traces_table']}" if cfg["traces_table"] else "")
          + " → 끝난 뒤 fab remote kick 으로 회수·채점, 결과는 fab sync")
    return key


def tfm_names(found) -> list[str]:
    """코드에서 찾은 이름 → TFM 이름 (KumoTabular → kumo)."""
    return sorted({m.lower().removesuffix("tabular") for m in found})


def tfm_imports(code: str) -> list[str]:
    """코드의 import 문이 부르는 TFM (fm_audit.WATCH 의 최상위 모듈)."""
    mods = []
    for frm, names in IMPORT_LINE.findall(code):
        mods += [frm] if frm else [n.split()[0] for n in names.split(",") if n.strip()]
    return sorted({WATCH[m.split(".")[0]] for m in mods if m.split(".")[0] in WATCH})


def code_text(name: str, raw: bytes) -> str:
    """작업 디렉터리 파일 → 코드 글. .ipynb 는 코드 셀 source 만 이어 붙임 (JSON 그대로면 줄이 따옴표로 시작해
    import 문 검사에 안 걸림). 깨진 노트북은 원문 그대로."""
    text = raw.decode("utf-8", "replace")
    if name.endswith(".ipynb"):
        try:
            cells = json.loads(text).get("cells", [])
            return "\n".join("".join(c["source"]) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                             for c in cells if c.get("cell_type") == "code")
        except (ValueError, AttributeError, KeyError, TypeError):
            return text
    return text


def log_commands(o, key=None):
    """하네스 기록 (JSON) 의 명령 칸 (COMMAND_KEYS) 글. Codex 는 item.command, Claude Code 는 Bash 도구의 input.command."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from log_commands(v, k)
    elif isinstance(o, list):
        for v in o:
            yield from log_commands(v, key)
    elif isinstance(o, str) and key in COMMAND_KEYS:
        yield o


def code_commands(code: str) -> list[str]:
    """코드 글의 pip 설치 줄 → 명령 글. 노트북 `!pip`·`%pip`, 파이썬 `os.system("pip install ...")` 꼴 (따옴표 안),
    `subprocess.run([sys.executable, "-m", "pip", "install", ...])` 꼴 (따옴표 안 낱말을 이어 붙임), 셸 스크립트 줄."""
    out = []
    for line in code.splitlines():
        words = re.findall(r"""(['"])(.*?)\1""", line)
        if any(w in ("pip", "pip3", "install") for _, w in words) and not PIP_INSTALL.search(line):
            line = " ".join(w for _, w in words)
        elif (quoted := [w for _, w in words if PIP_INSTALL.search(w)]):
            line = quoted[0]
        if PIP_INSTALL.search(line):
            out.append(line.lstrip().lstrip("!%"))
    return out


def _shell_split(text: str) -> list[str]:
    """셸 글 → 낱말 (연산자 && ; | 와 리다이렉트는 따로). 따옴표가 안 맞으면 공백으로만 나눔."""
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        return list(lex)
    except ValueError:
        return text.split()


def _words(cmd: str) -> list[str]:
    """셸 명령 → 낱말, 줄 끝은 ';'. bash -lc "..." (Codex 의 명령) 는 안쪽을 풀고, heredoc (python - <<'PY' ... PY) 의
    본문은 명령이 아니라서 뺌."""
    words = _shell_split(cmd)
    if len(words) >= 3 and words[0].rsplit("/", 1)[-1] in ("bash", "sh", "zsh") and re.fullmatch(r"-\w*c\w*", words[1]):
        return [*_words(words[2]), *words[3:]]
    out = []
    for line in HEREDOC.sub(" ; ", cmd).replace("\\\n", " ").splitlines():  # 줄 이음 (\ 줄바꿈) 은 한 줄로
        out += [*_shell_split(line), ";"]
    return out


def pip_target(cmds: list[str]) -> list[str]:
    """하네스 환경 밖에 깐 패키지 (pip freeze 전·후 비교 = pip_added 에 안 나옴): pip install 의 --target/-t/--prefix/--root,
    또는 실행 중 만든 가상환경 (python -m venv·virtualenv·uv venv <폴더>) 의 pip·python 으로 깐 것. 이름은 소문자·`-`,
    버전·extra 는 뗌. 실행한 명령만 봄 (설치가 성공했는지는 보지 않음). 2026-11-14 TabArena hazelnut × Codex 가
    `pip install --target /tmp/tabpfn_modules ...` 로 tabpfn·tabicl 을 깔고 PYTHONPATH 로 씀 (#18)."""
    venvs = {m.rstrip("/") for c in cmds for m in VENV_MAKE.findall(c)}
    found = set()
    for c in cmds:
        if not PIP_INSTALL.search(c):
            continue
        words = _words(c)
        # 같은 명령 안에서 새 가상환경을 켠 뒤 (source <폴더>/bin/activate) 의 pip 는 그 환경에 깜
        activated = any(re.search(rf"(?:\bsource|(?:^|[;&|(]\s*)\.)\s+{re.escape(v)}/bin/activate\b", c, re.M)
                        for v in venvs)
        for i, w in enumerate(words):
            if w != "install" or i == 0 or words[i - 1].rsplit("/", 1)[-1] not in ("pip", "pip3"):
                continue
            # 설치한 인터프리터: pip 앞 낱말 (… -m pip → python 경로, uv pip → uv, /tmp/v/bin/pip → 그 경로)
            exe = words[i - 3] if i >= 3 and words[i - 2] == "-m" else words[i - 1]
            outside = activated or any(exe.startswith(f"{v}/") for v in venvs)
            names, j = [], i + 1
            while j < len(words) and words[j] not in (";", "&&", "||", "|", "&"):
                w2 = words[j]
                # 리다이렉트: shlex 가 2>&1 을 '2' '>&' '1' 로, > log 를 '>' 'log' 로 나눔 → 연산자와 그 대상을 건너뜀
                if j + 1 < len(words) and w2.isdigit() and re.match(r"[<>]", words[j + 1]):
                    j += 1
                    continue
                if re.match(r"\d*[<>&]*[<>]", w2):
                    j += 2 if re.fullmatch(r"\d*(?:>>?|<|&>|>&)", w2) else 1
                    continue
                opt = w2.split("=", 1)[0]
                if opt in PIP_OUTSIDE_OPTS or re.fullmatch(r"-t\S+", w2):  # -t/tmp/x (값을 붙여 씀) 도
                    outside = True
                if opt in ("--python", "-p"):  # uv pip install --python <새 가상환경>
                    val = w2.split("=", 1)[1] if "=" in w2 else (words[j + 1] if j + 1 < len(words) else "")
                    outside |= any(val == v or val.startswith(f"{v}/") for v in venvs)
                if opt in PIP_VALUE_OPTS and "=" not in w2:
                    j += 2
                    continue
                if not w2.startswith("-") and not re.match(r"[./~$]|\w+://", w2):
                    names.append(re.split(r"[<>=!~\[;@ ]", w2, maxsplit=1)[0])
                j += 1
            if outside:
                found |= {n.lower().replace("_", "-") for n in names if re.fullmatch(r"[A-Za-z0-9][\w.-]*", n)}
    return sorted(found)


def submission_origin(d: Path, mtime: float | None) -> float | None:
    """제출 내용이 처음 생긴 시각: workspace.tgz 안에서 submission.csv 와 바이트가 같은 파일의 가장 이른 mtime (초 단위,
    tar 는 정수 초), 없으면 mtime (실행기가 잰 제출 파일 시각)."""
    sub, tgz = d / "submission.csv", d / "workspace.tgz"
    if mtime is None or not sub.exists() or not tgz.exists():
        return mtime
    data, times = sub.read_bytes(), [mtime]
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.size == len(data) and tar.extractfile(m).read() == data:
                times.append(float(m.mtime))
    return min(times)


def tfm_runtime(d: Path, meta: dict) -> dict:
    """fm_audit.jsonl (실행 중 TFM import 기록, runner.py 의 install_audit) → 실제로 import 한 TFM, 그런 프로세스 수,
    제출 내용이 처음 생긴 시각에 살아 있던 (import 뒤 ~ 끝) 프로세스가 import 한 TFM (tfm_at_submit). 제출 내용이 처음 생긴
    시각 = 작업 디렉터리에서 submission.csv 와 내용이 같은 파일 중 가장 이른 수정 시각 (에이전트는 흔히 sub6.csv 를 만든 뒤
    cp 로 제출함: 2026-11-12 nomad Claude Code, cp 시각에는 TFM 프로세스가 이미 끝남), 없으면 submission_mtime. 끝 = exit 줄,
    없으면 (강제 종료·os._exit) 마지막 alive + HEARTBEAT, 그것도 없으면 import + HEARTBEAT.
    tfm_at_submit 은 근사: TFM 예측을 파일로 저장하고 다른 프로세스에서 섞어 제출하면 안 잡힘, 오래 사는 프로세스가 TFM 을
    import 만 하고 제출은 다른 모델로 하면 잡힘 → 최종 풀이 코드로 확인. 이 기록 이전 실행은 모두 None
    (CSV 에서는 None 과 빈 목록이 둘 다 빈칸 → 기록이 있었는지는 tfm_runtime_procs 로: 빈칸 = 기록 없음, 0 = 안 씀)."""
    rows = read_jsonl(d / "fm_audit.jsonl")
    if not rows and (not meta.get("fm_audit") or str(meta["fm_audit"]).startswith("error")):
        return dict(tfm_runtime=None, tfm_runtime_procs=None, tfm_at_submit=None)
    imports = [r for r in rows if r.get("event") == "import"]
    last = {}  # pid → 마지막으로 살아 있던 시각
    for r in rows:
        last[r["pid"]] = max(last.get(r["pid"], 0), r["t"])
    exits = {r["pid"]: r["t"] for r in rows if r.get("event") == "exit"}
    sub_t = submission_origin(d, meta.get("submission_mtime"))
    at_submit = {r["tfm"] for r in imports if sub_t is not None
                 and r["t"] <= sub_t <= exits.get(r["pid"], last[r["pid"]] + HEARTBEAT)}
    return dict(tfm_runtime=sorted({r["tfm"] for r in imports}), tfm_runtime_procs=len({r["pid"] for r in imports}),
                tfm_at_submit=sorted(at_submit))


def posthoc(d: Path) -> dict:
    """회수한 기록에서: TFM 사용 흔적, Kaggle 풀이·토론 URL, 토큰 사용량, 429 대기.
    TFM 은 이렇게 나눔: 작업 디렉터리 코드에 이름이 있음(tfm_in_code) / 코드에 import 문(tfm_imported) / 실행 중 실제로
    import 됨(tfm_runtime·tfm_runtime_procs·tfm_at_submit, tfm_runtime 참고) / 새로 설치함(pip_added, 환경 밖에 깐 것은
    pip_target) / 대화·명령 기록에 언급만(tfm_mentions). +fm 실행은 지시문에 TFM 이름이 있어 tfm_mentions 가 늘 차므로 보지 않음. fm_offered 는 +fm 이
    깔아 준 TFM (사전 점검 통과).
    (2026-11-09 Codex: TabPFN·AutoGluon 이 깔렸는지 importlib 로 확인만 하고 안 씀 → 언급만으로 '사용'이라 하면 틀림)"""
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    code, cmds = [], []
    if (d / "workspace.tgz").exists():
        with tarfile.open(d / "workspace.tgz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith((".py", ".ipynb", ".sh")):
                    code.append(code_text(m.name, tar.extractfile(m).read()))
    logs = [p.read_text(errors="replace") for p in d.glob("*.jsonl")
            if p.name not in ("proxy.jsonl", "otel.jsonl", "fm_audit.jsonl")]
    for text in logs:
        for line in text.splitlines():
            if PIP_INSTALL.search(line):  # 줄마다 JSON 을 다 풀지 않게 (기록이 수 MB)
                try:
                    cmds += log_commands(json.loads(line))
                except ValueError:
                    pass
    code_blob, blob = "\n".join(code), "\n".join(code + logs)
    cmds += code_commands(code_blob)
    before = set((d / "pip_before.txt").read_text().splitlines()) if (d / "pip_before.txt").exists() else set()
    after = set((d / "pip_after.txt").read_text().splitlines()) if (d / "pip_after.txt").exists() else set()
    calls = read_jsonl(d / "proxy.jsonl")
    known = {m for _, m, _ in CONFIGS.values()} | {a for *_, a in CONFIGS.values()} | {None}

    def tok(*ks):
        return sum(c.get(k, 0) for c in calls for k in ks)

    return dict(tfm_in_code=tfm_names(TFM_PATTERN.findall(code_blob)),
                tfm_mentions=tfm_names(TFM_PATTERN.findall(blob)),
                tfm_imported=tfm_imports(code_blob), **tfm_runtime(d, meta),
                fm_offered=sorted(k for k, v in (meta.get("fm_warmup") or {}).items() if v.get("ok")),
                pip_added=sorted(after - before) if (d / "pip_after.txt").exists() else None,  # 없음 = 끝까지 안 감 (run_crashed)
                pip_target=pip_target(cmds),  # 환경 밖 (--target·--prefix·새 가상환경) 에 깐 것. pip_added 에 안 나옴
                kaggle_urls=len(KAGGLE_PATTERN.findall(blob)),
                data_source_urls=len(DATA_SOURCE_PATTERN.findall(blob)),
                retries_429=sum(c.get("retries_429", 0) for c in calls),
                waited_429=round(sum(c.get("waited", 0) for c in calls), 1),  # 흔들기로 소수 → 합의 부동소수 잡음 제거
                llm_calls=len(calls), llm_errors=sum(1 for c in calls if (c.get("status") or 0) >= 400),
                # Anthropic 의 input_tokens 는 캐시분을 뺀 값이라 캐시 읽기·쓰기를 더함 (OpenAI 는 캐시 포함해서 셈)
                # runner.py 의 INPUT_TOKEN_KEYS·OUTPUT_TOKEN_KEYS 와 같은 규칙 (trace 요약과 맞게, 바꾸면 둘 다)
                input_tokens=tok("input_tokens", "prompt_tokens", "cache_read_input_tokens",
                                 "cache_creation_input_tokens"),
                output_tokens=tok("output_tokens", "completion_tokens"),
                other_models=sorted({str(c["asked"]) for c in calls if c.get("asked") not in known}))


def node_columns(meta: dict) -> dict:
    """meta.json 의 노드 사양 (kaggle-master#32) → 표 열. CPU·메모리는 잰 값(호스트·affinity·cgroup 한도) 중 가장 작은 값:
    컨테이너가 실제로 쓸 수 있는 양은 그중 작은 쪽 (cgroup 한도가 호스트보다 크게 잡혀 있을 수도 있음)."""
    n = meta.get("node") or {}

    def smallest(*keys):
        vals = [v for k in keys if isinstance(v := n.get(k), (int, float)) and v > 0]
        return min(vals) if vals else None

    gpu = n.get("gpu")
    return dict(node_cpus=smallest("cgroup_cpus", "cpu_affinity", "cpu_count"),
                node_mem_gb=smallest("cgroup_mem_gb", "mem_total_gb", "mem_phys_gb"),
                node_tmp_free_gb=n.get("disk_free_gb:/tmp"),
                node_gpu="; ".join(gpu) if isinstance(gpu, list) else gpu)


def fill_defaults(t: pd.DataFrame) -> pd.DataFrame:
    """mlebench_runs.csv 에 나중에 생긴 열의 옛 행 값 (그 열 이전 실행의 실제 값): llm_route=fm_api, 이어 가기 0 번·
    첫 종료 = harness_seconds. grade_run (append_run) 과 recheck_table 이 같이 씀."""
    if "llm_route" in t:
        t["llm_route"] = t["llm_route"].fillna("fm_api")
    if "continues" in t:
        t["continues"] = t["continues"].fillna(0).astype(int)
        t["first_exit_seconds"] = t["first_exit_seconds"].fillna(t["harness_seconds"])
    if "run_crashed" in t:  # 이 열 이전 실행: 실행기가 죽은 실행은 exclude 로 뺐음
        t["run_crashed"] = t["run_crashed"].eq(True)  # 빈칸·False → False (fillna 는 object 열에서 경고)
    return t


def read_json(path: Path) -> dict:
    """JSON 파일 → dict. 없거나 쓰다 죽어 잘린 파일은 {} (정리 작업이 같은 오류로 계속 실패하지 않게)."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def recover_from_progress(d: Path) -> bool:
    """실행기가 마지막 회수 (meta.json) 전에 죽은 실행 (컨테이너가 사라짐): progress/ 의 사본으로 회수 폴더를 채움
    (runner.sync_progress). 기록 (*.jsonl)·pip_before.txt 는 그대로, meta 는 마지막 회차의 것, 제출은 submission.json 의
    sha256 과 맞을 때만 (쓰다 죽은 csv 는 안 씀), workspace.tgz 는 열리면 → 그 제출로 채점 (시간 초과 실행처럼 그때까지
    저장한 제출). 죽은 때 돌던 구간까지 이어 가기로 셈 (segments_started), timed_out·exit_code 는 None (끝까지 안 감).
    pip_after.txt 가 없어 pip_added 는 빈칸 (posthoc). meta 의 run_crashed 로 표에 남김. 회수할 게 없으면 False."""
    prog = d / "progress"
    if (d / "meta.json").exists() or not prog.is_dir():
        return False
    for f in [*prog.glob("*.jsonl"), prog / "pip_before.txt", prog / "full_instructions.txt"]:
        if f.exists() and not (d / f.name).exists():
            shutil.copy(f, d / f.name)
    meta = read_json(prog / "meta.json")
    sub, info = prog / "submission.csv", read_json(prog / "submission.json")
    if info and sub.exists() and hashlib.sha256(sub.read_bytes()).hexdigest() == info.get("sha256"):
        shutil.copy(sub, d / "submission.csv")
        meta.update({k: info[k] for k in ("submission_from", "submission_mtime", "format_ok", "format_msg") if k in info})
    if (tgz := prog / "workspace.tgz").exists():
        try:
            with tarfile.open(tgz) as tar:
                tar.getmembers()  # 끝까지 읽혀야 씀 (쓰다 죽은 사본이면 사후 검사가 깨짐)
            shutil.copy(tgz, d / "workspace.tgz")
        except (OSError, tarfile.TarError, EOFError):
            pass
    started = meta.get("segments_started") or len(meta.get("segments") or []) + 1  # 죽을 때 늘 한 구간이 돌고 있었음
    meta.update(run_crashed=True, continues=started - 1, timed_out=None, exit_code=None)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return True


def run_table(info: dict) -> Path:
    """실행이 들어갈 표: TabArena 실행 (제출 기록의 bench) 은 tabarena_runs.csv, 나머지는 mlebench_runs.csv."""
    return TA_TABLE if info.get("bench") == "tabarena" else TABLE


def append_run(row: dict, table: Path = TABLE):
    """채점 행 하나를 표에 (옛 행의 새 열은 fill_defaults 로 채움)."""
    with table_lock(table):
        t = pd.concat([pd.read_csv(table), pd.DataFrame([row])]) if table.exists() else pd.DataFrame([row])
        fill_defaults(t).to_csv(table, index=False)


def grade_run(key: str, info: dict, report: dict | None) -> dict:
    """회수한 실행 하나의 표 행. report 는 원격 채점 보고서 (submission.csv 가 없으면 None)."""
    d = AGENT_RUNS / key
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    ta = info.get("bench") == "tabarena"
    bench = dict(comp=info["comp"], anon_id=info["cid"]) if ta else dict(mlebench=info["cid"])
    row = dict(timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
               **bench, config=info["config"], harness=info["harness"], model=info["model"],
               rep=info["rep"], hours=info["hours"], key=key, llm_route=llm_route(info), timed_out=meta.get("timed_out"),
               harness_seconds=meta.get("harness_seconds"), format_ok=meta.get("format_ok"), **node_columns(meta),
               # 이어 가기: 횟수와 첫 종료 시각. 이 기능 이전 실행(meta 에 없음)은 이어 가지 않았으므로 0 과 harness_seconds
               continues=meta.get("continues", 0),
               first_exit_seconds=meta.get("first_exit_seconds", meta.get("harness_seconds")),
               run_crashed=bool(meta.get("run_crashed")))  # 실행기가 중간에 죽어 progress/ 사본으로 채점 (recover_from_progress)
    keys = ["score", "metric_error", "valid_submission", "error"] if ta else \
        ["score", "valid_submission", "any_medal", "gold_medal", "above_median", "lb_rank", "lb_teams"]
    if report is not None:
        row.update({k: report.get(k) for k in keys})
    else:
        row.update(score=None, valid_submission=False)
    p = posthoc(d)
    row.update({k: (" ".join(v) if isinstance(v, list) else v) for k, v in p.items()})
    append_run(row, run_table(info))
    return row


def recheck_table():
    """회수해 둔 기록으로 mlebench_runs.csv·tabarena_runs.csv 의 사후 검사 열을 다시 계산 (검사 규칙을 고친 뒤). 원격 정리 작업이 부름."""
    for table in (TABLE, TA_TABLE):
        if table.exists():
            recheck_one(table)


def recheck_one(table: Path):
    with table_lock(table):
        t = pd.read_csv(table)
        # 회수 기록이 있는 행만 (2026-11-10 이전 실행의 기록은 kaggle-master 로컬에 있고 Volume 에 없음 → 그 행은 그대로)
        have = t.index[[(AGENT_RUNS / key).is_dir() for key in t.key]]
        checks = pd.DataFrame([{k: " ".join(v) if isinstance(v, list) else v
                                for k, v in posthoc(AGENT_RUNS / t.at[i, "key"]).items()} for i in have], index=have)
        for c in checks:
            t[c] = t[c].astype(object) if c in t else None
            t.loc[have, c] = checks[c]
        print(f"사후 검사 다시 계산: {len(have)}/{len(t)}행 (나머지는 회수 기록이 없어 그대로)")
        fill_defaults(t).to_csv(table, index=False)
    cols = ["mlebench", "comp", "config", "score", "tfm_in_code", "tfm_imported", "tfm_runtime", "tfm_at_submit", "pip_added", "pip_target"]
    print(t[[c for c in cols if c in t]].to_string(index=False))  # 회수 기록이 하나도 없으면 새 열이 아직 없음


def runs(main_cfg) -> dict[str, dict]:
    """Volume 의 제출 기록 {key: manifest + 처리 표시}. state/agent/ 를 한 번에 받아 읽음 (기록마다 CLI 를 부르지 않게)."""
    from fab.remote import dbx

    with tempfile.TemporaryDirectory() as d:
        res = dbx(main_cfg, "fs", "cp", "-r", f"dbfs:{main_cfg['volume']}/state/agent", d, check=False, timeout=600)
        if res.returncode:
            return {}
        files = {p.name: p for p in Path(d).glob("*.json")}
        out = {}
        for name, p in sorted(files.items()):
            if name.endswith((".done.json", ".attempts.json", ".progress.json")):
                continue
            key = name.removesuffix(".json")
            done = files.get(f"{key}.done.json")
            out[key] = dict(json.loads(p.read_text()), done=json.loads(done.read_text()) if done else None)
        return out


def peek(keys: list[str] | None = None):
    cfg = settings()
    for key, info in runs(remote_settings()).items():
        if (keys and key not in keys) or (not keys and info["done"]) or "run_id" not in info:
            continue
        with tempfile.TemporaryDirectory() as d:
            got = dbx(cfg, "fs", "cp", "-r", f"dbfs:{info['out']}/progress", d, check=False)
            if got.returncode:
                print(f"{key}: 진행 기록 없음 (아직 하네스 시작 전이거나 이 기능 이전 실행)")
                continue
            elapsed = (Path(d) / "elapsed.txt").read_text().strip() if (Path(d) / "elapsed.txt").exists() else "?"
            calls = read_jsonl(Path(d) / "proxy.jsonl")
            status = {}
            for c in calls:
                status[c.get("status")] = status.get(c.get("status"), 0) + 1
            errs = [c.get("proxy_error") or c.get("error") for c in calls if (c.get("status") or 0) >= 400]
            # trace 전송 (traces_table): 출처(fab = 프록시 LLM span, harness = 하네스 OTLP)별 상태 수. 403 = SP 표 권한 없음
            otel = {}
            for c in read_jsonl(Path(d) / "otel.jsonl"):
                k = f"{c.get('source', '?')}:{c.get('status')}"
                otel[k] = otel.get(k, 0) + 1
            print(f"{key}: {elapsed}s, LLM {len(calls)}회 {status}, 429 대기 {round(sum(c.get('waited', 0) for c in calls), 1)}s"
                  + (f", trace 전달 {otel}" if otel else "")
                  + (f"\n  마지막 오류: {str(errs[-1])[:300]}" if errs else ""))


def main(argv: list[str], bench: str | None = None):
    """bench=tabarena (fab ta agent): run 의 대상은 TabArena r0f0 대회 이름 (ta-<데이터셋>). 에이전트에게는 설정의 익명 id
    (anon_id) 만 보임: 원격 실행 이름·Volume 의 데이터 묶음·run.json. 대회 이름은 내 Volume 의 제출 기록에만."""
    ta = bench == "tabarena"
    ap = argparse.ArgumentParser(prog="fab ta agent" if ta else "fab mleb agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="에이전트 실행 제출")
    r.add_argument("cid", help="TabArena r0f0 대회 이름 (ta-<데이터셋>)" if ta else "MLE-bench 대회 id")
    r.add_argument("configs", nargs="+", choices=list(CONFIGS))
    r.add_argument("--rep", type=int, default=1, help="반복 번호 (같은 설정을 여러 번 돌릴 때 이름 구분)")
    r.add_argument("--hours", type=float, default=1.0)
    r.add_argument("--continue", dest="cont", action="store_true",
                   help="별도 설정(+cont): 시간이 남았는데 하네스가 스스로 끝나면 같은 세션을 이어 감 (kaggle-master#35)")
    r.add_argument("--fm", action="store_true",
                   help="C2 (+fm): 환경에 TFM(tabpfn_v2·tabicl·kumo)을 깔고 지시문에 사용 예를 더함 (fab/agent/fm_notes.txt)")
    sub.add_parser("list", help="제출한 실행 목록")
    k = sub.add_parser("peek", help="실행 중인 실행의 진행 기록 (2분마다 갱신): LLM 호출 수·상태·429·프록시 오류")
    k.add_argument("keys", nargs="*", help="없으면 채점 안 된 실행 전부")
    sub.add_parser("recheck", help="사후 검사 열 다시 계산 요청 (검사 규칙을 고친 뒤, 원격 정리 작업이 처리)")
    x = sub.add_parser("exclude", help="실행을 결과에서 빼는 요청 (실행 환경 문제 등). 원격 정리 작업이 사유를 남기고 mlebench_runs.csv 행 삭제")
    x.add_argument("keys", nargs="+")
    x.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run" and ta:
        from fab.bench.tabarena import ensure_prepared
        from fab.bench.tabarena.remote import upload_answers

        check_continue(args.configs, args.cont)
        settings()
        comp, anon = ta_target(args.cid)
        ensure_prepared(comp)
        upload_answers(remote_settings(), [comp.name])  # 채점용 정답 (내 Volume)
        for c in args.configs:
            submit(anon, c, args.rep, args.hours, args.cont, args.fm, comp=comp)
    elif args.cmd == "run":
        from fab.bench.mlebench.remote import prepare_grading

        check_continue(args.configs, args.cont)
        settings()  # 제출 전에 설정 확인 (SP·scratch Volume)
        prepare_grading(remote_settings(), [args.cid])  # 끝난 뒤 원격에서 채점할 재료 (리더보드·정답·채점 코드), 설정 수와 무관하게 한 번
        for c in args.configs:
            submit(args.cid, c, args.rep, args.hours, args.cont, args.fm)
    elif args.cmd == "recheck":
        submit_item(remote_settings(), "recheck", f"{datetime.now():%Y%m%d-%H%M%S}", {})
        print("사후 검사 다시 계산 요청 → fab remote kick, 결과는 fab sync")
    elif args.cmd == "peek":
        peek(args.keys or None)
    elif args.cmd == "exclude":
        main_cfg = remote_settings()
        known = items(main_cfg, "agent")  # 한 번만 조회
        if unknown := [k for k in args.keys if k not in known]:
            raise SystemExit(f"제출 기록에 없는 실행: {', '.join(unknown)} (fab mleb agent list)")
        submit_item(main_cfg, "exclude", f"{datetime.now():%Y%m%d-%H%M%S}", dict(keys=args.keys, reason=args.reason))
        print(f"제외 요청: {', '.join(args.keys)} ({args.reason}) → fab remote kick")
    else:
        for key, info in runs(remote_settings()).items():
            done = info["done"]
            state = "pending" if not done else f"excluded ({done['excluded']})" if done.get("excluded") else \
                f"error ({done.get('error', '')[-80:]})" if done.get("status") == "error" else \
                f"graded score={done.get('score')}"
            print(f"{key}  run_id={info.get('run_id')}  {state}")
