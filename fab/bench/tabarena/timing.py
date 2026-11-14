"""원격 채점 시간 재기 (#20). 정리 작업 (fab/remote/reconcile.py 의 timing 항목) 안에서 같은 (대회, 모델) 을 두 경로로 채점해 잼.
진짜 결과표는 안 바뀜: 표는 Volume 의 tmp/timing/<key>/ 에 쓰고, 끝나면 로컬 표를 되돌림.

    uv run --frozen fab ta timing hazelnut-spread-contaminant-detection --models tabpfn_v2 tabicl kumo   # → fab remote kick

  순서: after → before → after2 (Volume 읽기 캐시 때문에 뒤에 도는 쪽이 빠를 수 있어 after 를 앞뒤로)
  before  2026-11-14 까지의 경로를 그대로 다시 함: 대회마다 표 5개 읽기 (tables.begin) → 모델마다 test_info · make_submission
          (test.csv 읽기, 제출 csv 를 Volume 에 쓰고 앞부분 출력) → Grader.grade (정답 csv·json 읽기, 제출 csv 읽기) → grade_row
          (test_info 다시) → append_rows (tabarena.csv 읽고 씀) → 표 5개를 Volume 에 (tables.flush) → 진행 기록 쓰기
  after   지금 경로 (fab.bench.tabarena.grade): 제출 파일 없이 test.npy 로 바로, 정답·test id 는 대회마다 한 번, 표는 끝에 한 번
두 경로의 점수가 같은지도 봄 (같은 (대회, 모델) 의 score 차이). FM 결과 합치기 (reconcile 의 fm: 실행 출력을 outputs/ 로
복사) 는 그 데이터셋의 FM 실행 하나의 results/ 를 scratch 로 shutil.copytree 의 copyfile (지금) → copy2 (예전) → copyfile 순으로 잼.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd


def _before(comps, models, note, grader, tables, prog: Path) -> tuple[list[dict], dict]:
    from fab.bench.tabarena import GRADES, grade_row
    from fab.core.competition import load_competition
    from fab.core.results import append_rows, test_info
    from fab.submission import make_submission

    rows, t = [], {}
    graded = []
    for name in comps:
        t0 = time.perf_counter()
        tables.begin()
        comp = load_competition(name)
        crow = []
        for m in models:
            if test_info(name, m) is None:
                continue
            report = grader.grade(name, make_submission(name, [m]))
            if report.get("valid_submission"):
                crow.append(grade_row(comp, m, report, note))
        t["grade"] = t.get("grade", 0.0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        if crow:
            append_rows(pd.DataFrame(crow), GRADES)
        tables.flush()
        graded.append(name)
        prog.write_text(json.dumps(graded))
        t["table"] = t.get("table", 0.0) + time.perf_counter() - t0
        rows += crow
    return rows, t


def measure(volume: Path, key: str, comps: list[str], models: list[str], grader, tables, merge_src: Path | None) -> dict:
    from fab.bench.tabarena import grade
    from fab.core.results import RESULTS
    from fab.remote import TABLES

    saved = {n: (RESULTS / n).read_bytes() if (RESULTS / n).exists() else None for n in TABLES}

    def restore():
        for n, data in saved.items():
            if data is None:
                (RESULTS / n).unlink(missing_ok=True)
            else:
                (RESULTS / n).write_bytes(data)

    scratch = volume / "tmp" / "timing" / key
    scratch.mkdir(parents=True, exist_ok=True)
    real, tables.remote = tables.remote, scratch
    out = dict(comps=len(comps), models=models)
    try:
        # 순서 효과 (Volume 읽기 캐시) 를 보려고 after → before → after
        for phase in ("after", "before", "after2"):
            restore()
            t0 = time.perf_counter()
            if phase == "before":
                before, tm = _before(comps, models, "timing", grader, tables, scratch / "progress.json")
            else:
                tm = {}
                after, _ = grade(comps, models, "timing", grader, timer=tm)
                t1 = time.perf_counter()
                tables.flush()
                tm["flush"] = time.perf_counter() - t1
            out[phase] = dict(seconds=round(time.perf_counter() - t0, 1), rows=len(before if phase == "before" else after),
                              **{k: round(v, 1) for k, v in tm.items()})
        a = {(r["comp"], r["model"]): r["score"] for r in before}
        b = {(r["comp"], r["model"]): r["score"] for r in after}
        out["same_keys"] = a.keys() == b.keys()
        out["max_abs_diff"] = max((abs(a[k] - b[k]) for k in a.keys() & b.keys()), default=None)
        out["identical"] = out["same_keys"] and all(a[k] == b[k] for k in a)
        if merge_src is not None and merge_src.is_dir():
            n = sum(1 for p in merge_src.rglob("*") if p.is_file())
            for name, fn in (("copyfile", shutil.copyfile), ("copy2", shutil.copy2), ("copyfile2", shutil.copyfile)):
                t0 = time.perf_counter()
                shutil.copytree(merge_src, scratch / f"merge_{name}", copy_function=fn, dirs_exist_ok=True)  # 매번 새 폴더
                out[f"merge_{name}"] = dict(seconds=round(time.perf_counter() - t0, 1), files=n)
    finally:
        restore()
        tables.remote = real
        shutil.rmtree(scratch, ignore_errors=True)
    return out
