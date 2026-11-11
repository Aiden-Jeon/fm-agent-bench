"""Databricks 에서 `fab run` 을 실행 (제출만). TFM 은 AI Runtime(서버리스 GPU, `databricks air run`),
GBDT(lgbm·xgb·catboost)는 GPU 를 안 쓰므로 CPU serverless 일회성 작업(`databricks jobs submit`)으로 따로.

예: uv run --frozen fab databricks spaceship-titanic lgbm catboost tabpfn_v2 tabicl
    uv run --frozen fab databricks allstate-claims-severity mitra_ft --split-folds   # fold 마다 GPU 작업 하나씩 동시에
    uv run --frozen fab databricks spaceship-titanic tabpfn_v2 --watch               # 제출 뒤 로그 보기 (끊어도 실행은 계속)

제출하면 Volume 의 state/fm/<key>.json 에 기록 → 원격 정리 작업(fab/remote/reconcile.py)이 끝난 실행의 예측·OOF 를
Volume 의 outputs/ 로 옮기고 results.csv 에 행을 추가 (--split-folds 는 묶음이 다 끝나면 합침). 결과표는 fab sync 로 받음.
→ 제출한 뒤 맥북을 꺼도 됨. 예측 배열은 로컬로 내려오지 않음.

준비: databricks CLI 로그인 + databricks.local.yaml (환경별 설정이라 git 에서 제외) + fab remote deploy (정리 작업)
    profile: <~/.databrickscfg 프로필>
    volume: /Volumes/<catalog>/<schema>/<volume>   # 대회 데이터·실행 결과·원격 상태를 둘 UC Volume
    accelerator: GPU_1xH100                         # 선택. 기본 GPU 종류 (--accelerator 가 우선)
    image: <catalog>.<schema>.km-air:v4             # 선택. 패키지를 미리 설치한 이미지 (docker/air). 없으면 실행마다 pip 설치

동작: 대회 데이터를 Volume 에 한 번 올려 두고 (없으면 로컬에서 kaggle 로 받아 업로드), 코드 스냅샷과 함께 제출.
     모델 가중치는 Volume 의 cache/models.tar 에 모아 두고 실행마다 다시 받지 않음.
     Kaggle·HF 토큰은 워크스페이스로 보내지 않음 (HF gated 가중치는 로컬에서 받아 Volume 의 weights/ 에 올림).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

from fab.core.competition import ROOT, load_competition
from fab.core.download import download
from fab.models import GATED, base_name
from fab.runner.common import CODE_PATHS, add_run_args, check_fold_args, pip_args, run_argv

LOCAL_CONFIG = ROOT / "databricks.local.yaml"
DEFAULT_ACCELERATOR = "GPU_1xH100"  # A10G(24GB)는 TFM 이 OOM 재시도를 반복해 느림
BASE_PIP = [["scikit-learn", "lightgbm", "xgboost", "catboost"]]  # AI Runtime 기본 환경에는 없음

# GPU 를 안 쓰는 모델 → CPU serverless 작업으로 따로 실행. GPU 노드에서 TFM 과 같이 돌리면 CPU 를 나눠 쓰느라
# 극도로 느림 (2026-10-23 8xH100: Restaurant 110행 xgb fold 하나 593s) → GPU 시간만 잡아먹음
CPU_MODELS = {"lgbm", "xgb", "catboost"}
CPU_ENV = "4"  # serverless environment version (Python 3.12)
CPU_PACKAGES = {"numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "lightgbm", "xgboost", "catboost"}

MODEL_CACHE = "cache/models.tar"  # Volume 안의 모델 가중치 캐시 (Hugging Face·TabPFN)

RUNNER = '''\
import contextlib, importlib.metadata, os, re, shutil, subprocess, sys, tarfile, time, uuid
from pathlib import Path

T0 = time.time()


def step(msg):
    print(f"[runner] {msg} ({time.time() - T0:.0f}s)", flush=True)


SRC = Path("/tmp/fab")  # code_source 를 쓰기 가능한 곳으로 복사 (folds.csv, results/ 생성)
shutil.copytree(Path(sys.argv[0]).parent, SRC)  # serverless 작업(CPU)에는 __file__ 이 없음
data = SRC / "competitions" / COMP / "data"
data.mkdir(parents=True, exist_ok=True)
for f in Path(DATA).iterdir():
    (data / f.name).symlink_to(f)


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_name(spec):  # "pkg[extra]>=1", "pkg @ git+...", "git+https://.../Repo.git@sha" → 배포 이름
    if spec.startswith("git+"):
        return spec.split("#")[0].rstrip("/").rsplit("/", 1)[-1].split("@")[0].removesuffix(".git")
    return re.split(r"[\\s\\[<>=!~@;]", spec, maxsplit=1)[0]


INSTALLED = {canon(d.metadata["Name"]) for d in importlib.metadata.distributions()}
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
for args in PIP:  # 모델별 설치 실패는 해당 모델만 실패하도록 계속 진행
    # 이미지에서는 이미 설치된 패키지는 건너뜀 (이미지를 다시 빌드하지 않고 추가한 모델만 설치).
    # 기본 환경에서는 -U 업그레이드 등 지정한 그대로 설치
    if IMAGE and all(canon(dist_name(a)) in INSTALLED for a in args if not a.startswith("-")):
        continue
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
(out / "pip_list.txt").write_text("\\n".join(sorted(f"{d.metadata['Name']}=={d.version}"
                                                   for d in importlib.metadata.distributions())) + "\\n")
step("패키지 준비")

# 모델 가중치: Volume 의 tar 를 로컬 디스크에 풀어 쓰고 (XDG_CACHE_HOME 아래 huggingface/, tabpfn/),
# fab run 이 성공하고 새로 받은 가중치가 있으면 tar 를 다시 만듦. 캐시는 있으면 좋은 것 → 읽기·쓰기 실패는 경고만.
# 동시에 끝난 실행끼리는 나중 것이 덮어씀 (빠진 가중치는 다음에 다시 받음)
cache = Path("/tmp/model_cache")
cache.mkdir(parents=True, exist_ok=True)
try:
    if CACHE and Path(CACHE).exists():  # CPU 작업은 가중치가 필요 없어 CACHE 없음
        with tarfile.open(CACHE) as tar:
            tar.extractall(cache, filter="fully_trusted")  # HF 캐시는 snapshots → blobs 심볼릭 링크
except (tarfile.TarError, OSError, EOFError) as e:
    print(f"[runner] 모델 캐시 읽기 실패 → 가중치를 새로 받음: {e!r}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)


def weights():  # .locks 와 덜 받은 파일(.incomplete)은 캐시에 넣지 않음
    return {p for d in ("huggingface", "tabpfn") for p in (cache / d).rglob("*")
            if p.is_file() and ".locks" not in p.parts and p.suffix != ".incomplete"}


def keep(t):
    return None if ".locks" in t.name or t.name.endswith(".incomplete") else t


before = weights()
step(f"모델 캐시 복원 ({len(before)}개 파일)")
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"])
env = {**os.environ, "XDG_CACHE_HOME": str(cache),
       "FAB_WEIGHTS": WEIGHTS,  # gated 가중치 위치 (fab/models/tfm.py gated_file)
       "FAB_REMOTE": "1"}  # fab run 은 원격에서만 (fab/runner/run.py)
code = subprocess.run([sys.executable, "-u", "-m", "fab", "run", COMP, *ARGS], cwd=SRC, env=env).returncode
step("fab run 종료")
if (SRC / "results").exists():
    shutil.copytree(SRC / "results", out / "results", dirs_exist_ok=True)
new = weights() - before
if code == 0 and new and CACHE:
    tmp = Path(f"{CACHE}.{uuid.uuid4().hex}.tmp")  # 컨테이너마다 PID 가 같을 수 있어 uuid
    try:
        Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp, "w") as tar:
            for d in ("huggingface", "tabpfn"):
                if (cache / d).exists():
                    tar.add(cache / d, arcname=d, filter=keep)
        os.replace(tmp, CACHE)
        step(f"모델 캐시 갱신 (새 파일 {len(new)}개)")
    except (tarfile.TarError, OSError) as e:
        print(f"[runner] 모델 캐시 갱신 실패 (결과에는 영향 없음): {e!r}", flush=True)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
print(f"fab run 종료 코드: {code}")
if code:  # serverless(CPU) 작업은 IPython 위에서 돌아 sys.exit(0) 도 실패로 처리됨
    sys.exit(code)
'''


def settings(args) -> dict:
    """profile·volume 은 databricks.local.yaml 에서만 (fab.remote.settings 와 같은 값). 제출 기록·결과·정리 작업이
    모두 그 Volume 을 보므로 명령마다 바꾸지 않음. accelerator·image 는 명령에서 덮어쓸 수 있음."""
    from fab.remote import settings as remote_settings

    cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
    cfg.update({k: v for k, v in dict(accelerator=args.accelerator, image=args.image).items() if v})
    cfg.update(remote_settings())  # profile·volume 이 없으면 여기서 멈춤 (아무것도 제출하기 전에)
    cfg.setdefault("accelerator", DEFAULT_ACCELERATOR)
    if str(cfg.get("image")).lower() == "none":  # --image none / YAML 의 None·null
        cfg["image"] = None
    return cfg


def databricks(profile: str, *args, capture=False, check=True, timeout=600):
    return subprocess.run(["databricks", *args, "-p", profile], check=check, text=True,
                          capture_output=capture, timeout=timeout)


def ensure_data(comp, profile: str, volume: str) -> str:
    """Volume 에 대회 데이터가 없으면 로컬에서 받아 업로드. Volume 경로를 반환."""
    remote = f"{volume}/data/{comp.name}"
    ls = databricks(profile, "fs", "ls", f"dbfs:{remote}", capture=True, check=False)
    if ls.returncode == 0 and comp.train_file in ls.stdout.splitlines():
        return remote
    if not (comp.data_dir / comp.train_file).exists():
        download(comp)
    print(f"데이터 업로드: {comp.data_dir} → {remote}")
    databricks(profile, "fs", "cp", "-r", "--overwrite", str(comp.data_dir), f"dbfs:{remote}", timeout=3600)
    return remote


def ensure_weights(models: list[str], profile: str, volume: str) -> str:
    """모델이 쓰는 HF gated 가중치가 Volume 에 없으면 로컬(HF 로그인)에서 받아 업로드. weights 폴더 경로를 반환."""
    remote = f"{volume}/weights"
    for repo, filename in dict.fromkeys(f for m in models for f in GATED.get(base_name(m), [])):
        dst = f"dbfs:{remote}/{repo}/{filename}"
        ls = databricks(profile, "fs", "ls", f"dbfs:{remote}/{repo}", capture=True, check=False)
        if ls.returncode == 0 and filename in ls.stdout.splitlines():  # 파일 경로 ls 는 실패하므로 폴더를 봄
            continue
        from huggingface_hub import hf_hub_download

        print(f"가중치 업로드: {repo}/{filename} → {remote}")
        databricks(profile, "fs", "mkdir", f"dbfs:{remote}/{repo}")  # 파일 cp 는 상위 폴더를 만들지 않음
        databricks(profile, "fs", "cp", "--overwrite", hf_hub_download(repo, filename), dst, timeout=3600)
    return remote


def snapshot(work: Path, comp: str, run_args: list[str], models: list[str], data: str, out: str, cache: str,
             preinstalled: bool, weights: str):
    code = work / "fab"
    for p in CODE_PATHS:
        src = ROOT / p
        if src.is_dir():
            shutil.copytree(src, code / p, ignore=shutil.ignore_patterns("__pycache__", "data"))
        else:
            shutil.copy(src, code / p)
    consts = dict(COMP=comp, ARGS=run_args, PIP=BASE_PIP + pip_args(models), IMAGE=preinstalled, DATA=data, OUT=out,
                  CACHE=cache, WEIGHTS=weights)
    (code / "air_runner.py").write_text("".join(f"{k} = {v!r}\n" for k, v in consts.items()) + "\n" + RUNNER)
    return code


def run_key(comp: str, tag: str, kind: str = "") -> str:
    """실행 이름 = MLflow run 이름·Volume 결과 폴더 이름."""
    return f"{tag + '-' if tag else ''}{comp}{'-' + kind if kind else ''}-{datetime.now():%Y%m%d-%H%M%S}"


def submit(cfg: dict, comp: str, data: str, run_args: list[str], models: list[str], accelerator: str, timeout: int,
           tag: str) -> str:
    n_gpus = re.match(r"GPU_(\d+)x", accelerator)
    if not n_gpus:
        raise SystemExit(f"accelerator 형식이 GPU_<n>x<type> 이 아님: {accelerator}")
    key = run_key(comp, tag)
    out = f"{cfg['volume']}/runs/{key}"

    # 이미지에는 패키지가 미리 설치돼 있음 (docker/air) → 러너가 이미 있는 패키지는 설치를 건너뜀
    image = cfg.get("image")
    work = Path(tempfile.mkdtemp())
    try:
        code = snapshot(work, comp, run_args, models, data, out, f"{cfg['volume']}/{MODEL_CACHE}", bool(image),
                        ensure_weights(models, cfg["profile"], cfg["volume"]))
        (work / "air.yaml").write_text(yaml.safe_dump({
            "experiment_name": f"fab-{comp}", "mlflow_run_name": key,
            "compute": {"accelerator_type": accelerator, "num_accelerators": int(n_gpus[1])},
            "environment": {"unity_catalog_image": image} if image else {"version": 6},
            "code_source": {"type": "snapshot", "snapshot": {"root_path": str(code)}},
            "command": "python $CODE_SOURCE_PATH/air_runner.py",
            "timeout_minutes": timeout, "max_retries": 0,  # 실패 시 재시도하면 같은 실험을 반복하므로 끔
        }, sort_keys=False))
        res = databricks(cfg["profile"], "air", "run", "-f", str(work / "air.yaml"), "-o", "json", capture=True)
    finally:
        shutil.rmtree(work)
    info = json.loads(res.stdout[res.stdout.index("{"):])["data"]  # 앞에 업로드 진행 메시지가 섞임
    return dict(key=key, run_id=info["run_id"], comp=comp, models=models, out=out, url=info["dashboard_url"])


def record(cfg: dict, run: dict):
    """제출 기록을 Volume 에 (fab remote kick 으로 정리 작업을 돌리면 처리). 실행마다 제출 직후 바로 부름
    → 뒤 제출이 실패해도 앞서 낸 실행은 기록이 남음."""
    from fab.remote import submit_item

    submit_item(cfg, "fm", run["key"], run)
    print(f"제출 완료: {run['key']} (run_id {run['run_id']})\n  {run['url']}")


def requirement_lines(names: set[str]) -> list[str]:
    """docker/air/requirements.txt 에서 names 패키지 줄 (GPU 이미지와 같은 버전). 하나라도 없으면 중단."""
    found = {}
    for line in (ROOT / "docker" / "air" / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        name = re.sub(r"[-_.]+", "-", re.split(r"[\s\[<>=!~@;]", line, maxsplit=1)[0]).lower()
        if name in names:
            found[name] = line
    if missing := names - found.keys():
        raise SystemExit(f"docker/air/requirements.txt 에 패키지 없음: {', '.join(sorted(missing))}")
    return list(found.values())


def submit_cpu(cfg: dict, comp: str, data: str, run_args: list[str], models: list[str], timeout: int,
               tag: str) -> str:
    """CPU serverless 일회성 작업으로 제출. 같은 러너를 쓰고 패키지는 작업 환경이 설치 (러너는 설치를 건너뜀).
    코드는 Workspace 에 올림 (`spark_python_task` 는 Workspace 파일 실행) → 끝난 뒤 정리 작업이 지움."""
    key = run_key(comp, tag, "cpu")
    out = f"{cfg['volume']}/runs/{key}"
    me = json.loads(databricks(cfg["profile"], "current-user", "me", "-o", "json", capture=True).stdout)["userName"]
    remote = f"/Workspace/Users/{me}/.fab/{key}"
    work = Path(tempfile.mkdtemp())
    try:
        code = snapshot(work, comp, run_args, models, data, out, cache="", preinstalled=True, weights="")
        databricks(cfg["profile"], "workspace", "import-dir", str(code), remote, capture=True, timeout=1200)
        (work / "job.json").write_text(json.dumps({
            "run_name": f"fab-{key}", "timeout_seconds": timeout * 60,
            "tasks": [{"task_key": "fab", "environment_key": "fab",
                       "spark_python_task": {"python_file": f"{remote}/air_runner.py"}}],
            "environments": [{"environment_key": "fab",
                              "spec": {"environment_version": CPU_ENV, "dependencies": requirement_lines(CPU_PACKAGES)}}],
        }))
        res = databricks(cfg["profile"], "jobs", "submit", "--no-wait", "--json", f"@{work / 'job.json'}", "-o",
                         "json", capture=True)
    except BaseException:
        databricks(cfg["profile"], "workspace", "delete", "--recursive", remote, capture=True, check=False)
        raise
    finally:
        shutil.rmtree(work)
    run_id = str(json.loads(res.stdout)["run_id"])
    run = cli_json(cfg["profile"], "jobs", "get-run", run_id)
    return dict(key=key, run_id=run_id, comp=comp, models=models, out=out, url=run["run_page_url"], cpu=True,
                code=remote)  # code: 정리 작업이 끝난 뒤 지움


def cli_json(profile: str, *args, tries: int = 10, poll: int = 30) -> dict:
    """상태 조회 (JSON). 토큰 갱신·일시적 5xx 같은 CLI 실패는 poll 초 간격으로 tries 번까지 재시도."""
    for i in range(1, tries + 1):
        try:
            res = databricks(profile, *args, "-o", "json", capture=True, check=False, timeout=120)
            if res.returncode == 0:
                return json.loads(res.stdout[res.stdout.index("{"):])
            err = res.stderr.strip()[-200:]
        except subprocess.TimeoutExpired:
            err = "120초 응답 없음"
        if i == tries:
            raise RuntimeError(f"`databricks {' '.join(args)}` {tries}번 실패: {err}")
        print(f"  {time.strftime('%H:%M:%S')} `databricks {' '.join(args[:2])}` 실패 ({i}/{tries}), 재시도: {err}")
        time.sleep(poll)


def status(profile: str, run_id: str) -> str:
    return cli_json(profile, "air", "get", run_id)["data"]["status"]


def wait(profile: str, run_id: str, poll: int = 30) -> str:
    while (s := status(profile, run_id)) in ("PENDING", "QUEUED", "BLOCKED"):
        print(f"  {time.strftime('%H:%M:%S')} {s}")
        time.sleep(poll)
    databricks(profile, "air", "logs", run_id, check=False, timeout=None)  # 끝날 때까지 로그 스트리밍
    while (s := status(profile, run_id)) in ("PENDING", "QUEUED", "BLOCKED", "RUNNING"):  # 스트림이 끊긴 경우
        time.sleep(poll)
    return s


def wait_cpu(profile: str, run_id: str, poll: int = 30) -> str:
    """CPU 작업은 로그 스트리밍이 없음 → 상태가 바뀔 때만 출력하고, 끝나면 로그(뒷부분)를 한 번에 출력."""
    last = None
    while True:
        run = cli_json(profile, "jobs", "get-run", run_id)
        state = run["state"]
        if state["life_cycle_state"] != last:
            last = state["life_cycle_state"]
            print(f"  {time.strftime('%H:%M:%S')} CPU {last}")
        if last in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(poll)
    if tasks := run.get("tasks"):  # 환경 준비에서 실패하면 task 실행이 없을 수 있음
        res = databricks(profile, "jobs", "get-run-output", str(tasks[0]["run_id"]), "-o", "json", capture=True,
                         check=False)
        output = json.loads(res.stdout) if res.returncode == 0 else {}
        print("\n".join(output.get(k) or "" for k in ("logs", "error", "error_trace")).strip())
    if state.get("state_message"):
        print(f"CPU 작업 상태 메시지: {state['state_message']}")
    return state.get("result_state", last)


def watch(profile: str, run: dict):
    """로그 보기 (결과 처리는 원격 정리 작업). 끊어도 실행은 계속."""
    if run.get("cpu"):
        print(f"최종 상태 (CPU): {wait_cpu(profile, run['run_id'])}")
    else:
        print(f"최종 상태 (GPU): {wait(profile, run['run_id'])}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="fab databricks")
    add_run_args(ap)
    ap.add_argument("--watch", action="store_true",
                    help="제출 뒤 로그를 봄 (끝날 때까지). 결과 처리는 어차피 원격 정리 작업이 함 → 끊어도 됨")
    ap.add_argument("--accelerator", help=f"GPU 종류 (GPU_1xA10, GPU_8xH100 등). 기본: {LOCAL_CONFIG.name} 의 "
                                          f"accelerator, 없으면 {DEFAULT_ACCELERATOR}")
    ap.add_argument("--image", help=f"실행 이미지 <catalog>.<schema>.<image>:<tag> ({LOCAL_CONFIG.name} 의 image 를 "
                                    "덮어씀). none 이면 기본 환경에서 pip 설치")
    ap.add_argument("--timeout", type=int, default=720, help="실행 시간 제한 (분)")
    ap.add_argument("--tag", default="", help="실행 이름 태그 (MLflow run 이름·결과 폴더에 붙음)")
    ap.add_argument("--split-folds", action="store_true",
                    help="GPU 모델을 fold 마다 작업 하나로 나눠 동시에 제출 (GPU 작업이 fold 수만큼). 다 끝나면 원격 정리 작업이 "
                         "fold 를 합쳐 results.csv 에 기록. 벽시계 시간은 줄지만 GPU 시간은 작업마다 준비 시간만큼 늘어남")
    args = ap.parse_args(argv)
    check_fold_args(ap, args)
    if args.split_folds and (args.fold is not None or args.max_folds or args.fold_batch):
        ap.error("--split-folds 는 --fold·--max-folds·--fold-batch 와 같이 쓸 수 없음")
    if args.fold is not None and not args.split_folds:
        ap.error("--fold 만 따로 내면 합칠 묶음이 없음 → --split-folds 사용")

    cfg = settings(args)
    comp = load_competition(args.comp)  # 설정 존재 확인
    cpu = [m for m in args.models if base_name(m) in CPU_MODELS]
    gpu = [m for m in args.models if m not in cpu]
    if args.split_folds:
        if not gpu:
            ap.error("--split-folds 는 GPU 모델용 (GBDT 는 CPU 작업 하나로 충분히 빠름)")
        if int(re.match(r"GPU_(\d+)x", cfg["accelerator"])[1]) > 1:  # fold 작업 수 × 장수만큼 GPU 를 잡음
            ap.error(f"--split-folds 는 GPU 1장 작업으로 (지금 {cfg['accelerator']} × fold {comp.cv['n_splits']}개)")
    data = ensure_data(comp, cfg["profile"], cfg["volume"])
    runs = []

    def add(run):  # 제출 직후 바로 기록
        record(cfg, run)
        runs.append(run)

    if gpu and args.split_folds:
        batch = f"{args.tag + '-' if args.tag else ''}{comp.name}-{datetime.now():%Y%m%d-%H%M%S}"
        print(f"GPU 작업 {comp.cv['n_splits']}개 제출 (fold 마다 {cfg['accelerator']}, 묶음 {batch}) — 워크스페이스 GPU 쿼터 확인")
        try:
            for k in range(comp.cv["n_splits"]):  # 같은 초에 제출해도 실행 이름이 겹치지 않게 태그에 fold 번호
                fold_args = copy.copy(args)
                fold_args.fold, fold_args.fold_batch = k, batch
                run = submit(cfg, comp.name, data, run_argv(fold_args, gpu), gpu, cfg["accelerator"], args.timeout,
                             f"{args.tag + '-' if args.tag else ''}fold{k}")
                add(dict(run, batch=batch, fold=k))
        except BaseException:
            # fold 가 다 모이지 않는 묶음 → 합치지 않음으로 표시 (채점 요청이 이 묶음을 계속 기다리지 않게).
            # 이미 낸 fold 실행은 기록이 있어 정리 작업이 Workspace 코드 정리 등은 그대로 함
            from fab.remote import now, put_json

            put_json(cfg, f"state/batch/{batch}.done.json",
                     dict(status="aborted", submitted_folds=[r["fold"] for r in runs], at=now()))
            print(f"묶음 {batch} 제출 중단 → 합치지 않음으로 기록 (제출된 fold: {[r['fold'] for r in runs]})")
            raise
    elif gpu:
        add(submit(cfg, comp.name, data, run_argv(args, gpu), gpu, cfg["accelerator"], args.timeout, args.tag))
    if cpu:
        add(submit_cpu(cfg, comp.name, data, run_argv(args, cpu), cpu, args.timeout, args.tag))
    print("끝난 뒤 fab remote kick → 정리 작업이 결과를 합침, fab sync 로 결과표 받기 (진행 확인: fab remote status)")
    if args.watch:
        try:
            for run in runs:
                watch(cfg["profile"], run)
        except KeyboardInterrupt:  # 작업은 Databricks 에서 계속 돎
            raise SystemExit("\n로그 보기 중단 (작업과 결과 처리는 원격에서 계속)")
