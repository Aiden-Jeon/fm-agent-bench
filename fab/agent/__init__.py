"""MLE-bench 트랙 B: 하네스 × LLM 에이전트를 Databricks 서버리스 GPU 에서 실행·회수·채점.

    uv run fab mleb agent run spaceship-titanic claude-code-sonnet          # 제출 (기다리지 않음)
    uv run fab mleb agent run nomad2018-predict-transparent-conductors aide-gpt --rep 2
    uv run fab mleb agent run spaceship-titanic claude-code-sonnet --continue  # 별도 설정 +cont (kaggle-master#35)
    uv run fab mleb agent fetch                                             # 끝난 실행 회수 → 채점 → results/mlebench_runs.csv
    uv run fab mleb agent list

실행 환경 (docs/plans/mle-bench.md, 2026-11-09 결정):
  - 작업은 전용 SP 신원으로 돎 (run_as). SP 권한은 scratch Volume 읽기·쓰기 + (모든 사용자처럼) FM API 조회.
    FM API 시스템 엔드포인트에는 ACL 을 걸 수 없어서, 실행기 프록시가 모델을 설정값으로 고정하고 요청을 기록함
  - 정답(private)은 로컬 mlebench 캐시에만 있음 → Volume 에는 public 만 올림
  - A10 1장, 벽시계 hours 시간. 원격 쪽은 fab/agent/runner.py
  - 기본은 공식 설정: 하네스가 스스로 끝나면 그대로 끝. --continue 는 별도 설정(<config>+cont)으로, 시간이 남았는데 끝나면
    같은 세션에 "남은 시간 N분, 계속 개선하라" 를 보내 이어 감 (세 하네스 같은 규칙, 2026-11-10 결정, kaggle-master#35)

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
import fcntl
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fab.core.results import OUTPUTS, RESULTS, table_lock

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
GATEWAY_SUFFIX = "+gw"  # gateway: true 로 돌린 실행의 꼬리표 (분당 한도·거부 조건이 FM API 와 달라 섞지 않게, llm_route 열과 같은 뜻)
HARNESS_VERSIONS = {"aide": "v6.3.3", "claude-code": "2.1.319", "codex": "0.167.0"}  # MLE-bench 의 AIDE 포크 태그
ACCELERATOR = "GPU_1xA10"
RUNS = OUTPUTS / "_mlebench" / "agents" / "runs.json"  # key → {run_id, cid, config, ...}
LOCAL_RUNS = OUTPUTS / "_mlebench" / "agents"  # 회수한 실행: <key>/
TABLE = RESULTS / "mlebench_runs.csv"  # 에이전트 실행별 채점 (git 추적)
HERE = Path(__file__).parent
# 사후 검사: 에이전트가 TFM 을 썼는지, Kaggle 풀이·토론을 봤는지 (기록에서 찾음)
# 단어 경계: "carte" 가 "Cartesian" 에 걸렸음 (2026-11-09 nomad Codex)
TFM_PATTERN = re.compile(r"\b(tabpfn|tabicl|kumo|autogluon|mitra|limix|exaone|tabdpt|carte)\b", re.I)
# databricks fs ls 가 경로 없음으로 실패할 때의 메시지 (그 밖의 실패는 일시적 오류로 봄)
MISSING = re.compile(r"no such (?:file|directory)|not found|does not exist|RESOURCE_DOES_NOT_EXIST", re.I)
KAGGLE_PATTERN = re.compile(r"kaggle\.com/(?:code|competitions/[^/\s\"']+/(?:discussion|code))", re.I)


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


def upload_data(cfg, cid: str) -> str:
    """public 데이터를 tar 하나로 올림 (파일이 많은 대회도 한 번에). 이미 있으면 그대로."""
    from fab.bench.mlebench import _driver

    remote = f"{cfg['volume']}/data/{cid}.tgz"
    if dbx(cfg, "fs", "ls", f"dbfs:{cfg['volume']}/data", check=False).stdout.count(f"{cid}.tgz"):
        return remote
    public = Path(_driver("paths", cid)["public"])
    if not public.is_dir() or not any(public.iterdir()):
        raise SystemExit(f"{public} 비어 있음 → 먼저 fab download (mlebench: {cid} 인 대회)")
    with tempfile.TemporaryDirectory() as d:
        tgz = Path(d) / f"{cid}.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            for p in sorted(public.iterdir()):
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


def load_runs() -> dict:
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def record(key: str, info: dict):
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS.parent / ".runs.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
        runs[key] = {**runs.get(key, {}), **info}
        tmp = RUNS.with_suffix(".tmp")
        tmp.write_text(json.dumps(runs, indent=2))
        os.replace(tmp, RUNS)


def check_continue(configs: list[str], cont: bool):
    """AIDE 는 자기 time_limit 까지 돌고 이어 갈 세션이 없음 → +cont 가 공식 설정과 같아서 거부.
    여러 설정을 한 번에 낼 때는 제출 전에 전부 확인 (앞 설정만 제출되고 중간에 멈추지 않게)."""
    if cont and (aide := [c for c in configs if CONFIGS[c][0] == "aide"]):
        raise SystemExit(f"--continue 는 AIDE 해당 없음 (일찍 끝나지 않음, 이어 갈 세션 없음): {', '.join(aide)}")


def submit(cid: str, config: str, rep: int, hours: float, cont: bool = False) -> str:
    from fab.bench.mlebench import MLEBENCH_DIR, ensure_lfs

    harness, model, alias = CONFIGS[config]
    check_continue([config], cont)
    cfg = settings()
    label = f"{config}{CONTINUE_SUFFIX if cont else ''}{GATEWAY_SUFFIX if cfg['gateway'] else ''}"
    ensure_lfs(cid)
    data = upload_data(cfg, cid)
    # 키는 Volume 경로·command_path 에 들어감 → '+' 불가 (jobs submit 이 거부, 2026-11-10). 설정 이름(label)은 +cont 그대로
    key = f"{cid}__{label.replace('+', '-')}__r{rep}__{datetime.now():%Y%m%d-%H%M%S}"
    out = f"{cfg['volume']}/runs/{key}"
    run = dict(key=key, cid=cid, harness=harness, model=model, alias=alias, hours=hours, data_tgz=data, out=out,
               harness_versions=HARNESS_VERSIONS, **{"continue": cont}, gateway=cfg["gateway"],
               upstream_model=gateway_model(model) if cfg["gateway"] else model, traces_table=cfg["traces_table"])
    with tempfile.TemporaryDirectory() as d:
        code = Path(d) / "code"
        code.mkdir()
        shutil.copy(HERE / "runner.py", code)
        shutil.copy(HERE / "additional_notes.txt", code)
        shutil.copy(MLEBENCH_DIR / "environment" / "instructions.txt", code)  # 공식 지시문 그대로 (경로는 실행기가 바꿈)
        shutil.copy(MLEBENCH_DIR / "environment" / "validate_submission.sh", code)
        (code / "run.json").write_text(json.dumps(run, indent=2))
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
            "timeout_seconds": int(hours * 3600) + 45 * 60,  # 하네스 시간 + 설치·회수 여유
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
    record(key, dict(run, run_id=run_id, config=label, rep=rep, submitted=datetime.now(timezone.utc).isoformat()))
    print(f"제출: {key} (run_id {run_id})" + (f" · LLM 경로 Unity Gateway ({run['upstream_model']})" if cfg["gateway"] else "")
          + (f" · trace → {cfg['traces_table']}" if cfg["traces_table"] else ""))
    return key


def posthoc(d: Path) -> dict:
    """회수한 기록에서: TFM 사용 흔적, Kaggle 풀이·토론 URL, 토큰 사용량, 429 대기.
    TFM 은 셋으로 나눔: 작업 디렉터리 코드에 있음(tfm_in_code) / 새로 설치함(pip_added) / 대화·명령 기록에 언급만(tfm_mentions).
    (2026-11-09 Codex: TabPFN·AutoGluon 이 깔렸는지 importlib 로 확인만 하고 안 씀 → 언급만으로 '사용'이라 하면 틀림)"""
    code = []
    if (d / "workspace.tgz").exists():
        with tarfile.open(d / "workspace.tgz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith((".py", ".ipynb", ".sh")):
                    code.append(tar.extractfile(m).read().decode("utf-8", "replace"))
    logs = [p.read_text(errors="replace") for p in d.glob("*.jsonl") if p.name not in ("proxy.jsonl", "otel.jsonl")]
    code_blob, blob = "\n".join(code), "\n".join(code + logs)
    before = set((d / "pip_before.txt").read_text().splitlines()) if (d / "pip_before.txt").exists() else set()
    after = set((d / "pip_after.txt").read_text().splitlines()) if (d / "pip_after.txt").exists() else set()
    calls = read_jsonl(d / "proxy.jsonl")
    known = {m for _, m, _ in CONFIGS.values()} | {a for *_, a in CONFIGS.values()} | {None}

    def tok(*ks):
        return sum(c.get(k, 0) for c in calls for k in ks)

    return dict(tfm_in_code=sorted({m.lower() for m in TFM_PATTERN.findall(code_blob)}),
                tfm_mentions=sorted({m.lower() for m in TFM_PATTERN.findall(blob)}),
                pip_added=sorted(after - before), kaggle_urls=len(KAGGLE_PATTERN.findall(blob)),
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
    첫 종료 = harness_seconds. fetch (append) 와 recheck 가 같이 씀."""
    if "llm_route" in t:
        t["llm_route"] = t["llm_route"].fillna("fm_api")
    if "continues" in t:
        t["continues"] = t["continues"].fillna(0).astype(int)
        t["first_exit_seconds"] = t["first_exit_seconds"].fillna(t["harness_seconds"])
    return t


def append_run(row: dict):
    """채점 행 하나를 mlebench_runs.csv 에 (옛 행의 새 열은 fill_defaults 로 채움)."""
    with table_lock(TABLE):
        t = pd.concat([pd.read_csv(TABLE), pd.DataFrame([row])]) if TABLE.exists() else pd.DataFrame([row])
        fill_defaults(t).to_csv(TABLE, index=False)


def grade_run(key: str, info: dict) -> dict:
    from fab.bench.mlebench import _driver, ensure_lfs

    d = LOCAL_RUNS / key
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    row = dict(timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
               mlebench=info["cid"], config=info["config"], harness=info["harness"], model=info["model"],
               rep=info["rep"], hours=info["hours"], key=key, llm_route=llm_route(info), timed_out=meta.get("timed_out"),
               harness_seconds=meta.get("harness_seconds"), format_ok=meta.get("format_ok"), **node_columns(meta),
               # 이어 가기: 횟수와 첫 종료 시각. 이 기능 이전 실행(meta 에 없음)은 이어 가지 않았으므로 0 과 harness_seconds
               continues=meta.get("continues", 0),
               first_exit_seconds=meta.get("first_exit_seconds", meta.get("harness_seconds")))
    if (d / "submission.csv").exists():
        ensure_lfs(info["cid"])
        r = _driver("grade", info["cid"], str(d / "submission.csv"))
        row.update({k: r.get(k) for k in ["score", "valid_submission", "any_medal", "gold_medal", "above_median",
                                          "lb_rank", "lb_teams"]})
    else:
        row.update(score=None, valid_submission=False)
    p = posthoc(d)
    row.update({k: (" ".join(v) if isinstance(v, list) else v) for k, v in p.items()})
    append_run(row)
    return row


def fetch(keys: list[str] | None = None):
    cfg = settings()
    for key, info in load_runs().items():
        if (keys and key not in keys) or info.get("graded") or info.get("excluded") or "run_id" not in info:
            continue
        run = json.loads(dbx(cfg, "jobs", "get-run", str(info["run_id"]), "-o", "json").stdout)
        state = run.get("state", {})
        if state.get("life_cycle_state") not in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            print(f"{key}: {state.get('life_cycle_state')}")
            continue
        d = LOCAL_RUNS / key
        d.mkdir(parents=True, exist_ok=True)
        got = dbx(cfg, "fs", "cp", "-r", "--overwrite", f"dbfs:{info['out']}", str(d), check=False, timeout=3600)
        if got.returncode:
            # 결과 폴더가 정말 없을 때만 (실행기가 회수 전에 죽음) 실패로 채점. 복사 오류(네트워크·CLI 시간 초과)면
            # 채점하지 않고 남겨 둠 → 다음 fetch 에서 다시 받음 (graded=True 로 굳으면 실제 제출을 영영 채점 못 함)
            ls = dbx(cfg, "fs", "ls", f"dbfs:{info['out']}", check=False)
            if ls.returncode == 0 or not MISSING.search(ls.stderr or ""):
                print(f"{key}: 결과 복사 실패 → 채점 보류, 다시 fetch ({(got.stderr or '').strip()[:200]})")
                continue
            print(f"{key}: {state.get('result_state')} — 결과 폴더 없음 ({state.get('state_message', '')[:200]})")
        row = grade_run(key, info)
        record(key, dict(graded=True, result_state=state.get("result_state")))
        print(f"{key}: {state.get('result_state')} score={row.get('score')} timed_out={row.get('timed_out')} "
              f"tfm_code={row.get('tfm_in_code') or '-'} tfm_mentions={row.get('tfm_mentions') or '-'} "
              f"kaggle_urls={row.get('kaggle_urls')} 429={row.get('retries_429')}/{row.get('waited_429')}s "
              f"tokens={row.get('input_tokens')}/{row.get('output_tokens')} "
              f"node={row.get('node_cpus')}cpu/{row.get('node_mem_gb')}GB/tmp {row.get('node_tmp_free_gb')}GB "
              f"{row.get('node_gpu')}")


def peek(keys: list[str] | None = None):
    cfg = settings()
    for key, info in load_runs().items():
        if (keys and key not in keys) or (not keys and (info.get("graded") or info.get("excluded"))) \
                or "run_id" not in info:
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


def main(argv: list[str]):
    ap = argparse.ArgumentParser(prog="fab mleb agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="에이전트 실행 제출")
    r.add_argument("cid", help="MLE-bench 대회 id")
    r.add_argument("configs", nargs="+", choices=list(CONFIGS))
    r.add_argument("--rep", type=int, default=1, help="반복 번호 (같은 설정을 여러 번 돌릴 때 이름 구분)")
    r.add_argument("--hours", type=float, default=1.0)
    r.add_argument("--continue", dest="cont", action="store_true",
                   help="별도 설정(+cont): 시간이 남았는데 하네스가 스스로 끝나면 같은 세션을 이어 감 (kaggle-master#35)")
    f = sub.add_parser("fetch", help="끝난 실행 회수 → 채점")
    f.add_argument("keys", nargs="*")
    sub.add_parser("list", help="제출한 실행 목록")
    k = sub.add_parser("peek", help="실행 중인 실행의 진행 기록 (2분마다 갱신): LLM 호출 수·상태·429·프록시 오류")
    k.add_argument("keys", nargs="*", help="없으면 채점 안 된 실행 전부")
    sub.add_parser("recheck", help="회수해 둔 기록으로 mlebench_runs.csv 의 사후 검사 열을 다시 계산 (검사 규칙을 고친 뒤)")
    x = sub.add_parser("exclude", help="실행을 결과에서 뺌 (실행 환경 문제 등). runs.json 에 사유를 남기고 mlebench_runs.csv 행 삭제")
    x.add_argument("keys", nargs="+")
    x.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        check_continue(args.configs, args.cont)
        for c in args.configs:
            submit(args.cid, c, args.rep, args.hours, args.cont)
    elif args.cmd == "fetch":
        fetch(args.keys or None)
    elif args.cmd == "recheck":
        with table_lock(TABLE):
            t = pd.read_csv(TABLE)
            checks = pd.DataFrame([{k: " ".join(v) if isinstance(v, list) else v
                                    for k, v in posthoc(LOCAL_RUNS / key).items()} for key in t.key], index=t.index)
            t = pd.concat([t.drop(columns=[c for c in checks if c in t]), checks], axis=1)
            fill_defaults(t).to_csv(TABLE, index=False)
        print(t[["mlebench", "config", "score", "tfm_in_code", "tfm_mentions", "pip_added"]].to_string(index=False))
    elif args.cmd == "peek":
        peek(args.keys or None)
    elif args.cmd == "exclude":
        if unknown := [k for k in args.keys if k not in load_runs()]:
            raise SystemExit(f"runs.json 에 없는 실행: {', '.join(unknown)} (fab mleb agent list)")
        for key in args.keys:
            record(key, dict(excluded=args.reason))
        if TABLE.exists():
            with table_lock(TABLE):
                t = pd.read_csv(TABLE)
                t[~t.key.isin(args.keys)].to_csv(TABLE, index=False)
        print(f"제외: {', '.join(args.keys)} ({args.reason})")
    else:
        for key, info in load_runs().items():
            state = f"excluded ({info['excluded']})" if info.get("excluded") else \
                "graded" if info.get("graded") else "pending"
            print(f"{key}  run_id={info.get('run_id')}  {state}")
