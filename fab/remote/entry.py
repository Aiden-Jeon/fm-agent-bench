"""정리 작업 진입점 (Databricks serverless CPU, job `fab-reconcile`, `fab remote kick` 으로 실행). fab 을 import 하기 전 준비만 함.

`fab remote deploy` 가 Workspace 에 올림. 인자: <volume>.
Volume 의 fab/code.tgz (로컬 명령마다 지금 코드로 다시 올라감) 를 풀고 fab.remote.reconcile.main 실행.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path

volume = sys.argv[1]
src = Path("/tmp/fab")
shutil.rmtree(src, ignore_errors=True)
with tarfile.open(f"{volume}/fab/code.tgz") as tar:
    tar.extractall(src, filter="data")
os.environ["FAB_OUTPUTS"] = f"{volume}/outputs"  # 예측·OOF 는 Volume 에 바로 (fab.core.results.OUTPUTS)
os.environ["FAB_REMOTE"] = "1"
os.chdir(src)
sys.path.insert(0, str(src))

from fab.remote.reconcile import main  # noqa: E402  (sys.path·환경 변수를 잡은 뒤)

main(volume)
