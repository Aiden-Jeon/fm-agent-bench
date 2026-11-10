"""원격 결과표 → 로컬 results/*.csv (git 추적). 원격(Volume 의 tables/)이 원본, 로컬은 사본.

    uv run --frozen fab sync                                  # 결과표만 (몇 KB)
    uv run --frozen fab sync --pull outputs/spaceship-titanic/catboost   # 필요한 실행 출력만 골라 받기 → results/ 아래

결과표에 로컬보다 행이 적으면 (원격이 로컬 기록을 모름) 덮어쓰지 않고 멈춤 → --force 로만.
예측 배열은 기본으로 받지 않음 (로컬 메모리·디스크를 쓰지 않도록). --pull 은 Volume 의 상대 경로를 그대로 results/ 아래로.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from fab.core.results import OUTPUTS, RESULTS
from fab.remote import TABLES, dbx, list_names, settings


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="fab sync")
    ap.add_argument("--pull", nargs="*", default=[], metavar="PATH",
                    help="Volume 의 outputs/ 아래 경로를 골라 받음 (예: outputs/<대회>/<모델>, outputs/_mlebench/agents/<key>)")
    ap.add_argument("--force", action="store_true", help="원격 표가 로컬보다 행이 적어도 덮어씀")
    args = ap.parse_args(argv)
    cfg = settings()

    have = set(list_names(cfg, "tables"))
    if not have:
        raise SystemExit(f"{cfg['volume']}/tables 없음 → fab remote deploy (로컬 결과표를 초기값으로 올림)")
    with tempfile.TemporaryDirectory() as d:
        for name in TABLES:
            if name not in have:
                continue
            tmp = Path(d) / name
            dbx(cfg, "fs", "cp", f"dbfs:{cfg['volume']}/tables/{name}", str(tmp))
            new = pd.read_csv(tmp)
            local = RESULTS / name
            old = len(pd.read_csv(local)) if local.exists() else 0
            if len(new) < old and not args.force:
                print(f"{name}: 원격 {len(new)}행 < 로컬 {old}행 → 덮어쓰지 않음 (로컬에만 있는 행 확인 후 --force)")
                continue
            local.write_bytes(tmp.read_bytes())
            print(f"{name}: {old} → {len(new)}행" + (f" (+{len(new) - old})" if len(new) > old else ""))

    for rel in args.pull:
        rel = rel.strip("/")
        if not rel.startswith("outputs/"):
            raise SystemExit(f"--pull 은 outputs/ 아래 경로: {rel}")
        dst = OUTPUTS / rel.removeprefix("outputs/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        res = dbx(cfg, "fs", "cp", "-r", "--overwrite", f"dbfs:{cfg['volume']}/{rel}", str(dst), check=False,
                  timeout=3600)
        print(f"{rel} → {dst}" if res.returncode == 0 else f"{rel}: 받기 실패 {res.stderr.strip()[:200]}")
