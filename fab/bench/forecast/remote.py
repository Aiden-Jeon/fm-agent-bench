"""forecast 채점: 로컬은 정답을 내 Volume 에 올리고, 채점은 원격 정리 작업 (fab/remote/reconcile.py) 안에서.

Volume (databricks.local.yaml 의 volume, 내 신원) 자리:
    forecast/answers/<대회>.csv   정답 (id · timestamp · target · scale = 계절 오차) + <대회>.json (horizon · 분위수 · id 대응표)
에이전트는 다른 곳: SP 의 scratch Volume 의 data/<익명 id>-<해시>.tgz (public 만, fab/agent). TabArena 와 같은 격리.
채점은 fab/bench/forecast/metrics.py (fev 식을 numpy 로) 로 직접 → fev 가 필요 없음.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

REL = "forecast/answers"


def upload_answers(cfg, comps: list[str]):
    """정답 → 내 Volume 의 forecast/answers/ (늘 덮어씀: 파일이 작고, 설명·id 를 고쳐 다시 준비했을 때 옛 정답이 남지 않게)."""
    from fab.bench.forecast import ANSWERS
    from fab.remote import dbx

    files = [f"{c}.{ext}" for c in comps for ext in ("csv", "json")]
    if missing := [f for f in files if not (ANSWERS / f).exists()]:
        raise SystemExit(f"정답 없음: {missing[:5]} → fab fc prepare")
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            shutil.copy(ANSWERS / f, Path(d) / f)
        dbx(cfg, "fs", "mkdir", f"dbfs:{cfg['volume']}/{REL}")
        dbx(cfg, "fs", "cp", "-r", "--overwrite", d, f"dbfs:{cfg['volume']}/{REL}", timeout=1800)
    print(f"정답 업로드: 파일 {len(files)}개 → {cfg['volume']}/{REL}/")


class Grader:
    """정리 작업 안에서 채점. grade(대회, csv) → {score, metric_error, valid_submission, SQL, MASE, WQL, WAPE} 또는
    {valid_submission: False, error}."""

    def __init__(self, volume: Path):
        self.dir = Path(volume) / REL

    def info(self, comp: str) -> dict | None:
        import json

        f = self.dir / f"{comp}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def grade_frame(self, comp: str, sub) -> dict:
        """제출 표 (익명 id) → 채점 (FM 의 마지막 window 예측, fab.bench.forecast.collect)."""
        from fab.bench.forecast import read_answers, score_frame

        if (info := self.info(comp)) is None:
            return dict(valid_submission=False, error=f"정답 없음: {self.dir / f'{comp}.json'}")
        return score_frame(sub, read_answers(self.dir / f"{comp}.csv"), info)

    def grade(self, comp: str, csv) -> dict:
        from fab.bench.forecast import score_submission

        if (info := self.info(comp)) is None:
            return dict(valid_submission=False, error=f"정답 없음: {self.dir / f'{comp}.json'}")
        return score_submission(Path(csv), self.dir / f"{comp}.csv", info)
