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
from fab.models import GATED, base_name, is_refit
from fab.runner.common import CODE_PATHS, add_run_args, check_run_args, pip_args, run_argv

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
import contextlib, fcntl, importlib.metadata, json, os, re, shutil, subprocess, sys, tarfile, threading, time, uuid
from pathlib import Path

T0 = time.time()


def step(msg):
    print(f"[runner] {msg} ({time.time() - T0:.0f}s)", flush=True)


SRC = Path("/tmp/fab")  # code_source 를 쓰기 가능한 곳으로 복사 (folds.csv, results/ 생성)
shutil.copytree(Path(sys.argv[0]).parent, SRC)  # serverless 작업(CPU)에는 __file__ 이 없음
# JOBS = [(대회, Volume 데이터, fab run 인자), ...]: 보통 하나. TabArena 는 데이터셋의 split 대회들을 한 작업에서 차례로
for comp, data_path, _ in JOBS:
    data = SRC / "competitions" / comp / "data"
    data.mkdir(parents=True, exist_ok=True)
    for f in Path(data_path).iterdir():
        (data / f.name).symlink_to(f)


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_name(spec):  # "pkg[extra]>=1", "pkg @ git+...", "git+https://.../Repo.git@sha" → 배포 이름
    if spec.startswith("git+"):
        return spec.split("#")[0].rstrip("/").rsplit("/", 1)[-1].split("@")[0].removesuffix(".git")
    return re.split(r"[\\s\\[<>=!~@;]", spec, maxsplit=1)[0]


def find_uv():
    """uv 위치: PATH, 이 파이썬 옆, 흔한 설치 위치 (이미지에 따라 작업 프로세스의 PATH 밖일 수 있음)."""
    home = Path.home()
    cands = [shutil.which("uv"), str(Path(sys.executable).parent / "uv"), "/usr/local/bin/uv", "/usr/bin/uv",
             str(home / ".local" / "bin" / "uv"), str(home / ".cargo" / "bin" / "uv")]
    return next((c for c in cands if c and os.path.isfile(c) and os.access(c, os.X_OK)), None)


def installer():
    """(종류, 명령 앞부분) 또는 None. pip 가 있으면 pip: 워크스페이스의 패키지 저장소 설정(pip.conf·PIP_*)을 그대로 따름.
    UC 이미지(/opt/venv)는 uv 로 만든 venv 라 pip 가 없음 (2026-11-11, ase 설치가 'No module named pip' 로 실패)
    → 이미지에 있는 uv 로 이 파이썬에 설치."""
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        return "pip", [sys.executable, "-m", "pip", "install", "-q"]
    if uv := find_uv():
        cmd = [uv, "pip", "install", "-q", "--python", sys.executable]
        if sys.prefix == sys.base_prefix:  # venv 가 아니면 PEP 668 로 막힌 시스템 파이썬일 수 있음. 일회용 작업이라 허용
            cmd.append("--break-system-packages")
        return "uv", cmd
    return None


def uv_env():
    """uv 가 pip 의 패키지 저장소 설정을 따르게: PIP_* 환경 변수와 pip.conf 의 index-url·extra-index-url·trusted-host
    (워크스페이스 기본 패키지 저장소 등). 없으면 uv 기본값."""
    import configparser
    conf = {}
    for p in ["/etc/pip.conf", "/etc/xdg/pip/pip.conf", str(Path.home() / ".pip" / "pip.conf"),
              str(Path.home() / ".config" / "pip" / "pip.conf"), os.environ.get("PIP_CONFIG_FILE", "")]:
        if not (p and os.path.isfile(p)):
            continue
        c = configparser.ConfigParser()
        try:
            c.read(p)
        except configparser.Error:
            continue
        for sec in ("global", "install"):
            for k in ("index-url", "extra-index-url", "trusted-host"):
                if c.has_option(sec, k):
                    conf[k] = c.get(sec, k)
    env = dict(os.environ)
    for pip_key, uv_key, conf_key in (("PIP_INDEX_URL", "UV_INDEX_URL", "index-url"),
                                      ("PIP_EXTRA_INDEX_URL", "UV_EXTRA_INDEX_URL", "extra-index-url"),
                                      ("PIP_TRUSTED_HOST", "UV_INSECURE_HOST", "trusted-host")):
        if value := os.environ.get(pip_key) or conf.get(conf_key):
            env.setdefault(uv_key, " ".join(value.split()))  # pip 는 줄바꿈, uv 는 공백으로 여러 개
    return env


def uv_args(args):
    """pip 인자를 uv 로. pip 의 -U 는 필요한 것만 올리지만 uv 의 -U 는 의존성까지 다 올림 (numpy 가 오르면 kditransform 이
    깨짐, _LIMIX) → 이름 붙은 패키지만 올림 (--upgrade-package)."""
    if not any(a in ("-U", "--upgrade") for a in args):
        return list(args)
    names = [dist_name(a) for a in args if not a.startswith("-")]
    return [a for a in args if a not in ("-U", "--upgrade")] + [x for n in names for x in ("--upgrade-package", n)]


INSTALLED = {canon(d.metadata["Name"]) for d in importlib.metadata.distributions()}
out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
INSTALL, INSTALL_FAILED = ..., []  # ... = 아직 안 찾음 (찾아도 없으면 None, 한 번만 찾음)
for args in PIP:  # 모델별 설치 실패는 해당 모델만 실패하도록 계속 진행. 실패는 크게 출력 + out/install_failed.json
    # 이미지에서는 이미 설치된 패키지는 건너뜀 (이미지를 다시 빌드하지 않고 추가한 모델만 설치).
    # 기본 환경에서는 -U 업그레이드 등 지정한 그대로 설치
    if IMAGE and all(canon(dist_name(a)) in INSTALLED for a in args if not a.startswith("-")):
        continue
    if INSTALL is ...:
        INSTALL = installer()
    if INSTALL is None:
        reason = "pip·uv 둘 다 없음"
    else:
        kind, cmd = INSTALL
        res = subprocess.run([*cmd, *(uv_args(args) if kind == "uv" else args)], env=uv_env() if kind == "uv" else None)
        reason = None if res.returncode == 0 else f"{kind} 종료 코드 {res.returncode}"
    if reason:
        INSTALL_FAILED.append(dict(args=args, reason=reason))
        print(f"[runner] 설치 실패 ({reason}): {args} → 이 패키지를 쓰는 모델은 실패함", flush=True)
if INSTALL_FAILED:  # 정리 작업이 결과가 없을 때 이 파일을 보고 까닭을 남김 (작업은 SUCCESS 로 끝나도)
    (out / "install_failed.json").write_text(json.dumps(INSTALL_FAILED, ensure_ascii=False, indent=1))
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
SAVE_LOCK = threading.Lock()


def save(comp):
    """대회마다 그 대회의 출력 + results.csv 만 Volume 에 → 작업이 중간에 죽어도 끝난 대회는 남음 (전체를 매번 복사하지 않음).
    results.csv 는 다른 GPU 의 fab run 이 쓰는 중일 수 있어 fab 의 표 잠금을 잡고 복사. 잠금 파일 이름은
    fab/core/results.py 의 table_lock (.<표 이름>.lock) 과 같아야 함 (실행기는 fab 를 import 하지 않음)."""
    if len(JOBS) == 1:
        return
    if (SRC / "results" / comp).exists():
        shutil.copytree(SRC / "results" / comp, out / "results" / comp, dirs_exist_ok=True)
    if (SRC / "results" / "results.csv").exists():
        with SAVE_LOCK, open(SRC / "results" / ".results.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            (out / "results").mkdir(parents=True, exist_ok=True)
            shutil.copy(SRC / "results" / "results.csv", out / "results" / "results.csv")


def run_comp(comp, args, run_env, tag=""):
    """fab run 하나. tag 가 있으면 (GPU 여럿) 출력 줄마다 붙임."""
    cmd = [sys.executable, "-u", "-m", "fab", "run", comp, *args]
    if not tag:
        rc = subprocess.run(cmd, cwd=SRC, env=run_env).returncode
    else:
        proc = subprocess.Popen(cmd, cwd=SRC, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace")
        try:
            for line in proc.stdout:
                print(f"{tag} {line}", end="", flush=True)
        finally:  # 출력을 읽다 실패해도 자식이 끝나기 전에 같은 GPU 에 다음 대회를 띄우지 않음
            rc = proc.wait()
    step(f"fab run 종료: {comp} (코드 {rc}){' ' + tag if tag else ''}")
    save(comp)
    return rc


GPUS = []
if shutil.which("nvidia-smi"):
    GPUS = [ln.split(":")[0].split()[-1] for ln in subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                                                     text=True).stdout.splitlines() if ln.startswith("GPU ")]
codes = {}
if len(GPUS) > 1 and len(JOBS) > 1:
    # GPU 여럿 (GPU_8xH100 등) + 대회 여럿 (TabArena split): GPU 마다 작업자 하나, 남은 대회를 큰 것 (train.csv 크기) 부터 하나씩.
    # 작업자마다 GPU 하나 (CUDA_VISIBLE_DEVICES) · CPU 를 나눠 받음 (fab run 의 run_parallel 과 같은 까닭: 스레드가 전체 코어를 잡으면 느림)
    import queue
    todo = queue.Queue()
    for job in sorted(JOBS, key=lambda j: -sum(f.stat().st_size for f in Path(j[1]).glob("train*"))):
        todo.put(job)
    n = min(len(GPUS), len(JOBS))
    # CPU 수: affinity + cgroup v2 할당량 (fab/runner/common.py 의 cpu_count 와 같은 계산. 실행기는 fab 를 import 하지 않음)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    per = str(max(1, cpus // n))
    step(f"대회 {len(JOBS)}개를 GPU {n}장에 나눔 (작업자당 CPU 스레드 {per})")

    def worker(gpu):
        wenv = {**env, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": per, "FAB_THREADS": per}
        while True:
            try:
                comp, _, args = todo.get_nowait()
            except queue.Empty:
                return
            try:
                codes[comp] = run_comp(comp, args, wenv, f"[g{gpu}]")
            except Exception as e:  # 복사 실패 등: 그 대회만 실패로 (다른 대회는 계속)
                print(f"[runner] {comp} 실패: {e!r}", flush=True)
                codes[comp] = 1

    threads = [threading.Thread(target=worker, args=(g,)) for g in GPUS[:n]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
else:
    for comp, _, args in JOBS:  # 한 대회가 실패해도 나머지는 돌림
        codes[comp] = run_comp(comp, args, env)
code = next((codes.get(c, 1) for c, _, _ in JOBS if codes.get(c, 1)), 0)  # 종료 코드는 (JOBS 순서로) 처음 실패한 것
if len(JOBS) == 1 and (SRC / "results").exists():
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
    for entry in sorted(comp.data_dir.iterdir()):  # _features/ 등 _ 로 시작하는 로컬 캐시는 올리지 않음 (원격에서 새로 만듦)
        if not entry.name.startswith("_"):
            databricks(profile, "fs", "cp", *(["-r"] if entry.is_dir() else []), "--overwrite", str(entry),
                       f"dbfs:{remote}/{entry.name}", timeout=3600)
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
             preinstalled: bool, weights: str, jobs: list | None = None):
    """jobs: [(대회, Volume 데이터, fab run 인자), ...] 를 한 작업에서 차례로 (없으면 comp 하나). PIP 는 첫 대회 기준
    (피처 변형의 PIP 는 대회마다 같다고 봄: TabArena split 대회는 피처 변형이 없음)."""
    jobs = [list(j) for j in jobs] if jobs else [[comp, data, run_args]]
    code = work / "fab"
    wanted = {j[0] for j in jobs}
    base_ignore = shutil.ignore_patterns("__pycache__", "data")

    def ignore(d, names):  # TabArena 생성 split 대회 (tas-*, 816개까지) 는 이 작업이 돌리는 것만 넣음
        skip = set(base_ignore(d, names))
        if Path(d) == ROOT / "competitions":
            skip |= {n for n in names if n.startswith("tas-") and n not in wanted}
        return skip

    for p in CODE_PATHS:
        src = ROOT / p
        if src.is_dir():
            shutil.copytree(src, code / p, ignore=ignore)
        else:
            shutil.copy(src, code / p)
    consts = dict(JOBS=jobs, PIP=BASE_PIP + pip_args(models, jobs[0][0]), IMAGE=preinstalled, OUT=out, CACHE=cache,
                  WEIGHTS=weights)
    (code / "air_runner.py").write_text("".join(f"{k} = {v!r}\n" for k, v in consts.items()) + "\n" + RUNNER)
    return code


def run_key(comp: str, tag: str, kind: str = "") -> str:
    """실행 이름 = MLflow run 이름·Volume 결과 폴더 이름."""
    return f"{tag + '-' if tag else ''}{comp}{'-' + kind if kind else ''}-{datetime.now():%Y%m%d-%H%M%S}"


def submit(cfg: dict, comp: str, data: str, run_args: list[str], models: list[str], accelerator: str, timeout: int,
           tag: str, jobs: list | None = None) -> dict:
    """jobs 가 있으면 comp 는 실행 이름에만 쓰고 그 대회들을 차례로 (snapshot). 기록의 comps = 그 대회들."""
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
                        ensure_weights(models, cfg["profile"], cfg["volume"]), jobs)
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
    return dict(key=key, run_id=info["run_id"], comp=comp, models=models, out=out, url=info["dashboard_url"],
                **({"comps": [j[0] for j in jobs]} if jobs else {}))


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
               tag: str, jobs: list | None = None) -> dict:
    """CPU serverless 일회성 작업으로 제출. 같은 러너를 쓰고 패키지는 작업 환경이 설치 (러너는 설치를 건너뜀).
    코드는 Workspace 에 올림 (`spark_python_task` 는 Workspace 파일 실행) → 끝난 뒤 정리 작업이 지움."""
    key = run_key(comp, tag, "cpu")
    out = f"{cfg['volume']}/runs/{key}"
    me = json.loads(databricks(cfg["profile"], "current-user", "me", "-o", "json", capture=True).stdout)["userName"]
    remote = f"/Workspace/Users/{me}/.fab/{key}"
    work = Path(tempfile.mkdtemp())
    try:
        code = snapshot(work, comp, run_args, models, data, out, cache="", preinstalled=True, weights="", jobs=jobs)
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
                code=remote, **({"comps": [j[0] for j in jobs]} if jobs else {}))  # code: 정리 작업이 끝난 뒤 지움


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
    check_run_args(ap, args)
    if args.split_folds and (args.fold is not None or args.max_folds or args.fold_batch):
        ap.error("--split-folds 는 --fold·--max-folds·--fold-batch 와 같이 쓸 수 없음")
    if args.fold is not None and not args.split_folds:
        ap.error("--fold 만 따로 내면 합칠 묶음이 없음 → --split-folds 사용")
    if args.split_folds and (refit := [m for m in args.models if is_refit(m)]):  # 원격에서 fold 마다 거절됨 (노드가 뜬 뒤)
        ap.error(f"refit 모델 {refit} 은 fold 가 없음 → --split-folds 와 같이 못 씀")

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
