"""mle-bench 가상환경에서 도는 보조 스크립트. fab 을 import 하지 않음.

로컬: fab/bench/mlebench/__init__.py 가 `python -I` 로 부름 (paths·prepare).
원격: grade_runner.py 가 import 해서 grade 를 부름 (Databricks serverless CPU 작업, fab/bench/mlebench/remote.py).

  paths <id>          대회 데이터 위치 (zip 을 둘 곳, public, private) → JSON
  prepare <id> <zip>  받아 둔 Kaggle zip 으로 MLE-bench 분할을 만들고 체크섬 확인 → JSON
  grade <id> <csv>    제출 파일 채점 → 보고서 JSON (+ 원래 Kaggle 리더보드에서의 위치 lb_rank / lb_teams)

결과 JSON 은 stdout 마지막 줄 (mlebench 로그는 stderr).
FAB_MLEBENCH_DATA 가 있으면 그곳을 데이터 위치로 씀 (원격: 정답을 올려 둔 Volume). 없으면 mlebench 기본 캐시.
"""

import json
import os
import sys
from pathlib import Path

import mlebench.data as data
from mlebench.grade import grade_csv
from mlebench.registry import registry

if os.environ.get("FAB_MLEBENCH_DATA"):
    registry = registry.set_data_dir(Path(os.environ["FAB_MLEBENCH_DATA"]))


def paths(cid: str) -> dict:
    comp = registry.get_competition(cid)
    return dict(dir=str(comp.raw_dir.parent), public=str(comp.public_dir), private=str(comp.private_dir))


def prepare(cid: str, zip_path: str) -> dict:
    comp = registry.get_competition(cid)
    # Kaggle 다운로드는 fab 이 함 (mlebench 가 쓰는 kaggle<1.7 은 ~/.kaggle/access_token 을 못 읽음).
    # 나머지(압축 풀기·분할·체크섬 확인)는 mlebench 그대로 → 공개 리더보드 에이전트와 같은 train/test
    data.download_dataset = lambda **_: Path(zip_path)
    data.download_and_prepare_dataset(comp)
    return paths(cid)


def grade(cid: str, csv: str) -> dict:
    comp = registry.get_competition(cid)
    report = grade_csv(Path(csv), comp).to_dict()
    if report["score"] is not None:
        lb = data.get_leaderboard(comp)["score"]
        better = lb < report["score"] if report["is_lower_better"] else lb > report["score"]
        report.update(lb_rank=int(better.sum()) + 1, lb_teams=int(len(lb)))
    return report


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    print(json.dumps({"paths": paths, "prepare": prepare, "grade": grade}[cmd](*args), default=str))
