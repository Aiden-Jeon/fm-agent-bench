"""TabArena 채점: 로컬은 public·정답을 내 Volume 에 올리고, 채점은 원격 정리 작업(fab/remote/reconcile.py) 안에서.

Volume (databricks.local.yaml 의 volume, 내 신원) 자리:
    data/<대회>/                 public (train.csv·test.csv·sample_submission.csv·description.md). FM 학습 작업이 읽음
    tabarena/answers/<대회>.csv  정답 (test 행의 id·타깃) + <대회>.json (클래스·지표). 정리 작업만 읽음
에이전트는 다른 곳: SP 의 scratch Volume 의 data/<익명 id>.tgz (public 만, fab/agent). SP 는 내 Volume 스키마 권한이 없음
(docs/mle-bench.md 의 "정답 격리와 권한"). 채점은 fab 의 지표 (fab/core/metrics.py) 로 직접 → MLE-bench 같은 별도 venv 없음.

올리기는 split 이 많아 (FM 맞춤 확인 816개) 파일마다 CLI 를 부르지 않고, 없는 대회만 임시 폴더에 모아 `fs cp -r` 한 번.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from fab.core.competition import Competition


def _missing(cfg, rel: str, names: list[str]) -> set[str]:
    from fab.remote import list_names

    have = {n.rstrip("/") for n in list_names(cfg, rel)}
    return {n for n in names if n not in have}


def _upload_dir(cfg, local: Path, rel: str):
    from fab.remote import dbx

    dbx(cfg, "fs", "mkdir", f"dbfs:{cfg['volume']}/{rel}")
    dbx(cfg, "fs", "cp", "-r", "--overwrite", str(local), f"dbfs:{cfg['volume']}/{rel}", timeout=7200)


def upload_data(cfg, comps: list[Competition]) -> dict[str, str]:
    """public → Volume 의 data/<대회>/ (폴더가 없는 대회만, 폴더 안은 보지 않음: 대회마다 ls 하면 816번). {대회: Volume 경로}."""
    names = [c.name for c in comps]
    if todo := _missing(cfg, "data", names):
        with tempfile.TemporaryDirectory() as d:
            for c in comps:
                if c.name in todo:
                    dst = Path(d) / c.name
                    dst.mkdir()
                    for f in sorted(c.data_dir.iterdir()):
                        if f.is_file() and not f.name.startswith("_"):
                            shutil.copy(f, dst / f.name)
            print(f"데이터 업로드: 대회 {len(todo)}개 → {cfg['volume']}/data/")
            _upload_dir(cfg, Path(d), "data")
    return {n: f"{cfg['volume']}/data/{n}" for n in names}


def upload_answers(cfg, comps: list[str]):
    """정답 → 내 Volume 의 tabarena/answers/ (없는 대회만)."""
    from fab.bench.tabarena import ANSWERS

    files = [f"{c}.{ext}" for c in comps for ext in ("csv", "json")]
    if missing := [f for f in files if not (ANSWERS / f).exists()]:
        raise SystemExit(f"정답 없음: {missing[:5]}... → fab ta prepare")
    todo = _missing(cfg, "tabarena/answers", files)
    if not todo:
        return
    with tempfile.TemporaryDirectory() as d:
        for f in todo:
            shutil.copy(ANSWERS / f, Path(d) / f)
        print(f"정답 업로드: 파일 {len(todo)}개 → {cfg['volume']}/tabarena/answers/")
        _upload_dir(cfg, Path(d), "tabarena/answers")


class Grader:
    """정리 작업 안에서 채점. grade(대회, csv) → {score, metric_error, valid_submission} 또는 {valid_submission: False, error}
    (에이전트 제출 파일). answers(대회) → (정답 표, 정보): FM 채점 (fab.bench.tabarena.grade) 이 대회마다 한 번 읽음."""

    def __init__(self, volume: Path):
        self.dir = Path(volume) / "tabarena" / "answers"

    def answers(self, comp: str):
        import json

        from fab.bench.tabarena import read_answers

        f = self.dir / f"{comp}.json"
        if not f.exists():
            raise FileNotFoundError(f"정답 없음: {f}")
        info = json.loads(f.read_text())
        return read_answers(self.dir / f"{comp}.csv", info), info

    def grade(self, comp: str, csv) -> dict:
        from fab.bench.tabarena import score_frame

        try:
            ans, info = self.answers(comp)
        except FileNotFoundError as e:
            return dict(valid_submission=False, error=str(e))
        try:
            sub = pd.read_csv(csv)
        except Exception as e:  # noqa: BLE001
            return dict(valid_submission=False, error=f"읽기 실패: {e!r}")
        return score_frame(sub, ans, info)
