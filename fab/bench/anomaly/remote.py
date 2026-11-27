"""이상탐지 채점 재료 (내 Volume) 와 원격 채점기. TabArena (fab/bench/tabarena/remote.py) 와 같은 꼴.

Volume (databricks.local.yaml 의 volume, 내 신원) 자리:
    data/<대회>/                    public (train.csv · test.csv · sample_submission.csv · description.md). FM 작업이 읽음
    anomaly/answers/<대회>.csv      정답 (test 행의 id · label) + <대회>.json. 정리 작업만 읽음
    anomaly/train_labels/<대회>.csv train 행의 라벨. 깨끗한 context 진단 (+clean) 의 FM 작업만 읽음
    anomaly/train_labels/<대회>.la.csv  la 마다 고른 PU 라벨 (라벨을 쓰는 분류기 +la<k>, docs/anomaly.md "라벨이 있는 이상탐지")
에이전트는 다른 곳: SP 의 scratch Volume 의 data/<익명 id>-<해시>.tgz (public 만, fab/agent). SP 는 내 Volume 스키마 권한이 없음.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fab.bench.tabarena.remote import _missing, _upload_dir

ANSWERS_REL, TRAIN_LABELS_REL = "anomaly/answers", "anomaly/train_labels"


def _upload_files(cfg, src: Path, rel: str, files: list[str], what: str, always: list[str] | None = None):
    """Volume 에 없는 파일만 올림. always: 있어도 다시 올릴 파일 (내용이 늘 수 있는 것)."""
    if missing := [f for f in files if not (src / f).exists()]:
        raise SystemExit(f"{what} 없음: {missing[:5]}... → fab ad prepare")
    if not (todo := sorted(_missing(cfg, rel, files) | set(always or []))):
        return
    with tempfile.TemporaryDirectory() as d:
        for f in todo:
            shutil.copy(src / f, Path(d) / f)
        print(f"{what} 업로드: 파일 {len(todo)}개 → {cfg['volume']}/{rel}/")
        _upload_dir(cfg, Path(d), rel)


def upload_answers(cfg, comps: list[str]):
    from fab.bench.anomaly import ANSWERS

    _upload_files(cfg, ANSWERS, ANSWERS_REL, [f"{c}.{ext}" for c in comps for ext in ("csv", "json")], "정답")


def upload_train_labels(cfg, comps: list[str], la: bool = False):
    """la: 분류기 · 빼기 변형용 <대회>.la.csv 도 (fab ad sup prepare 가 만듦). la 파일은 **늘 다시 올림**: la 열이 늘 수 있어서
    (2026-11-27: 공식 대회에 la75, S3 대회에 la1 · la10. 기존 열은 그대로인 것을 prepare 가 확인) "없는 파일만" 으로는 옛 파일이 남음."""
    from fab.bench.anomaly import TRAIN_LABELS

    la_files = [f"{c}.la.csv" for c in comps] if la else []
    _upload_files(cfg, TRAIN_LABELS, TRAIN_LABELS_REL, [f"{c}.csv" for c in comps] + la_files, "train 라벨", always=la_files)


def train_labels_dir(cfg) -> str:
    return f"{cfg['volume']}/{TRAIN_LABELS_REL}"


class Grader:
    """정리 작업 안에서 채점. grade(대회, csv) → 보고서 (에이전트 제출 파일), answers(대회) → 정답 표 (FM 채점이 대회마다 한 번)."""

    def __init__(self, volume: Path):
        self.dir = Path(volume) / ANSWERS_REL

    def answers(self, comp: str):
        from fab.bench.anomaly import read_answers

        if not (f := self.dir / f"{comp}.csv").exists():
            raise FileNotFoundError(f"정답 없음: {f}")
        return read_answers(f)

    def grade(self, comp: str, csv) -> dict:
        from fab.bench.anomaly import score_submission

        if not (f := self.dir / f"{comp}.csv").exists():
            return dict(valid_submission=False, error=f"정답 없음: {f}")
        return score_submission(Path(csv), f)
