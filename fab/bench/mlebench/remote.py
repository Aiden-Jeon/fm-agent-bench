"""MLE-bench 채점: 로컬은 채점 재료를 Volume 에 올리고, 채점은 원격 정리 작업(fab/remote/reconcile.py) 안에서.

로컬 (채점 요청·에이전트 제출 때 prepare_grading):
- 리더보드 csv (메달 기준선·순위): ensure_lfs 로 받은 뒤 mlebench/ 패키지째 묶어 Volume 의 mlebench/code.tgz 로
- 정답(private): 내 Volume 의 mlebench/data/<cid>/prepared/private/ 에 한 번 올림. 에이전트 SP 의 scratch Volume 과
  다른 곳이고 SP 는 이 스키마 권한이 없음 (docs/mle-bench.md). 정답을 만드는 분할(fab download)은 로컬 mle-bench 에서
원격 (Grader): fab 과 패키지 버전이 달라(mlebench 는 pandas 2.2·numpy 2.0·sklearn 1.5) 별도 venv 를 만들어 driver.py 실행.
venv 는 정리 작업 한 번에 처음 채점할 때 한 번만 만듦.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# 로컬 mle-bench 가상환경과 같은 버전 (2026-11-10, uv pip list). Python 은 원격 3.12 / 로컬 3.11
GRADE_DEPS = ["appdirs==1.4.4", "diskcache==5.6.3", "numpy==2.0.2", "pandas==2.2.3", "py7zr==1.1.4", "pyyaml==6.0.3",
              "scikit-learn==1.5.2", "scipy==1.17.1", "tenacity==9.2.1", "tqdm==4.70.1"]
WORK = Path("/tmp/mleb_grade")


# --- 로컬 ------------------------------------------------------------------------------------------------------


def ensure_answers(cfg: dict, cid: str):
    """정답(private)이 Volume 에 없으면 로컬 mlebench 캐시에서 올림."""
    from fab.bench.mlebench import _driver
    from fab.remote import dbx

    remote = f"{cfg['volume']}/mlebench/data/{cid}/prepared/private"
    ls = dbx(cfg, "fs", "ls", f"dbfs:{remote}", check=False)
    if ls.returncode == 0 and ls.stdout.strip():
        return
    private = Path(_driver("paths", cid)["private"])
    if not private.is_dir() or not any(private.iterdir()):
        raise SystemExit(f"{private} 비어 있음 → 먼저 fab download (mlebench: {cid} 인 대회)")
    print(f"정답 업로드: {private} → {remote}")
    dbx(cfg, "fs", "mkdir", f"dbfs:{remote}")
    dbx(cfg, "fs", "cp", "-r", "--overwrite", str(private), f"dbfs:{remote}", timeout=3600)


def upload_code(cfg: dict):
    """driver.py + 로컬 mle-bench 의 mlebench/ 패키지 → Volume 의 mlebench/code.tgz."""
    from fab.bench.mlebench import DRIVER, MLEBENCH_DIR
    from fab.remote import dbx

    with tempfile.TemporaryDirectory() as d:
        tgz = Path(d) / "code.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(DRIVER, arcname="driver.py")
            tar.add(MLEBENCH_DIR / "mlebench", arcname="mle-bench/mlebench",
                    filter=lambda t: None if "__pycache__" in Path(t.name).parts else t)
        dbx(cfg, "fs", "mkdir", f"dbfs:{cfg['volume']}/mlebench")
        dbx(cfg, "fs", "cp", "--overwrite", str(tgz), f"dbfs:{cfg['volume']}/mlebench/code.tgz")


def prepare_grading(cfg: dict, cids):
    """채점에 필요한 것 (리더보드·정답·채점 코드) 을 Volume 에. 원격 채점이 나중에 쓰므로 제출할 때 미리."""
    from fab.bench.mlebench import ensure_lfs

    for cid in dict.fromkeys(cids):
        ensure_lfs(cid)
        ensure_answers(cfg, cid)
    upload_code(cfg)


# --- 원격 ------------------------------------------------------------------------------------------------------


class Grader:
    """정리 작업 안에서 채점. grade(cid, csv) → driver.grade 보고서, 실패하면 {error}."""

    def __init__(self, volume: Path):
        self.volume = Path(volume)
        self.py: Path | None = None

    def _setup(self):
        shutil.rmtree(WORK, ignore_errors=True)
        with tarfile.open(self.volume / "mlebench" / "code.tgz") as tar:
            tar.extractall(WORK, filter="data")
        # serverless 환경 Python 에는 ensurepip 가 없음 → pip 없는 venv 를 만들고 바깥 pip 로 그 venv 에 설치 (pip --python)
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(WORK / ".venv")], check=True)
        py = WORK / ".venv" / "bin" / "python"
        subprocess.run([sys.executable, "-m", "pip", "--python", str(py), "install", "-q", *GRADE_DEPS], check=True)
        self.py = py

    def _env(self) -> dict:
        # 작업 환경의 PYTHONPATH 등이 venv 에 섞이면 새 버전 pandas 를 import 할 수 있음 → PYTHON* 은 지우고 우리 것만
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        return dict(env, PYTHONPATH=str(WORK / "mle-bench"), FAB_MLEBENCH_DATA=str(self.volume / "mlebench" / "data"))

    def grade(self, cid: str, csv) -> dict:
        if self.py is None:
            self._setup()
        # cwd: mlebench 가 체크섬 캐시(cache/)를 cwd 에 만듦
        res = subprocess.run([str(self.py), str(WORK / "driver.py"), "grade", cid, str(csv)], cwd=WORK / "mle-bench",
                             env=self._env(), capture_output=True, text=True)
        if res.returncode:
            return {"error": (res.stderr or res.stdout)[-3000:]}
        return json.loads(res.stdout.strip().splitlines()[-1])
