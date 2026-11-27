"""원격 전용: 이상탐지 대회 하나에서 방법들을 차례로 → <대회>/<방법>/test.npy (test.csv 행 순서의 점수, 클수록 이상) + test.json,
results.csv 에 행. fab/runner/run.py 의 main 이 anomaly 대회를 여기로 넘김 (fab databricks 실행기가 `fab run` 을 부르므로).
출력 자리 · 꼴이 FM 실행과 같아 정리 작업의 fm 단계가 그대로 합침 (fab/remote/reconcile.py).

    fab run <대회> <방법...> --note <글> [--train-labels <Volume 경로>]

--train-labels: +clean 방법 (깨끗한 context 진단) 이 읽는 train 라벨 폴더 (내 Volume, remote.py). 라벨을 쓰는 분류기 (`_clf+la<k>`) 와
빼기 변형 (`+dropla<k>`: 그 라벨이 1 인 행을 context 에서 뺌) 은 같은 폴더의 <대회>.la.csv (id + la 마다 ADBench 와 같이 고른 PU 라벨
열 la<k>) 를 읽음.
TabPFN 가중치는 v2 (동의 없이 받음, methods.FM). 실행기가 모델 캐시 (Volume 의 cache/models.tar) 에 남김.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fab.bench.anomaly import ID_COL, LABEL_COL
from fab.bench.anomaly import methods as M
from fab.core.competition import load_competition
from fab.core.results import OUTPUTS, append_rows, clear_test_info, write_test_info


def trim_context(name: str, X: np.ndarray, seed: int, cache: dict) -> np.ndarray:
    """라벨 없는 context 정리 (+trim<k>x<n> · +iftrim<k>): 정리 점수 상위 k% 를 빼기를 n 번. 행 순서는 그대로 (IForest 처럼 행
    순서로 표본을 뽑는 방법이 원래 순서와 같게). 첫 번 점수는 k 와 상관없어 cache 로 같이 씀 (trim5 · 10 · 20 이 한 번만 점수)."""
    scorer, k, n = M.trim_spec(name)
    idx = np.arange(len(X))
    for r in range(n):
        key = (scorer, 0) if r == 0 else (scorer, k, r)
        if key not in cache:
            cache[key] = M.train_scores(scorer, X[idx], seed + r)
        s = cache[key]
        drop = np.argsort(-s, kind="stable")[: int(round(k / 100 * len(idx)))]
        idx = np.delete(idx, drop)
    return X[idx]


def la_labels(labels_dir: str | None, comp: str, ids: pd.Series, k: int, what: str) -> np.ndarray:
    """<대회>.la.csv 의 la<k> 열을 train.csv 행 순서로 (라벨 준 이상 1, 나머지 0)."""
    if not labels_dir:
        raise RuntimeError(f"{what} 는 --train-labels 가 필요")
    la = pd.read_csv(Path(labels_dir) / f"{comp}.la.csv").set_index(ID_COL)[f"la{k}"]
    return la.loc[ids].to_numpy().astype(int)


def main(argv: list[str]):
    ap = argparse.ArgumentParser(prog="fab run (anomaly)")
    ap.add_argument("comp")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--note", default="")
    ap.add_argument("--train-labels", default=None)
    args = ap.parse_args(argv)
    if not os.environ.get("FAB_REMOTE"):
        sys.exit(f"원격 전용 (로컬은 제출만): fab ad fm {args.comp} --methods {' '.join(args.methods)}")
    if bad := [m for m in args.methods if not M.is_known(m)]:
        sys.exit(f"알 수 없는 방법: {bad}")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")  # tabpfn_common_utils 의 사용 기록 전송을 끔

    comp = load_competition(args.comp)
    train = pd.read_csv(comp.data_dir / "train.csv")
    test = pd.read_csv(comp.data_dir / "test.csv")
    feats = [c for c in train.columns if c != ID_COL]
    if [c for c in test.columns if c != ID_COL] != feats:
        sys.exit(f"{comp.name}: train · test 의 열이 다름")
    X_train, X_test = train[feats].to_numpy(dtype=float), test[feats].to_numpy(dtype=float)
    seed = int(comp.anomaly["seed"])
    rows, failed, cache = [], [], {}
    for m in args.methods:
        t0 = time.time()
        try:
            X_fit, extra = X_train, ""
            if M.is_supervised(m):  # 라벨을 쓰는 분류기: PU 라벨 (라벨 준 이상 1, 나머지 0)
                y = la_labels(args.train_labels, comp.name, train[ID_COL], M.la_spec(m), "+la 분류기")
                s = M.clf_scores(m, X_train, y, X_test, seed)
                if s.shape != (len(X_test),) or not np.isfinite(s).all():
                    raise ValueError(f"{m}: 점수 모양 {s.shape} 또는 NaN/inf")
                extra = f" n_labeled={int(y.sum())}"
            elif M.is_clean(m):  # 깨끗한 context 진단: train 의 정상 행만 (train 라벨을 씀 → Elo 밖)
                if not args.train_labels:
                    raise RuntimeError("+clean 은 --train-labels 가 필요")
                lab = pd.read_csv(Path(args.train_labels) / f"{comp.name}.csv").set_index(ID_COL)[LABEL_COL]
                X_fit = X_train[lab.loc[train[ID_COL]].to_numpy() == 0]
            elif M.trim_spec(m):  # 라벨 없는 정리 (E3): 비지도 그대로
                X_fit = trim_context(m, X_train, seed, cache)
            elif (k := M.drop_spec(m)) is not None:  # L1: 분류기와 같은 la 라벨이 1 인 행을 빼고 남은 행 (행 순서 그대로)
                y = la_labels(args.train_labels, comp.name, train[ID_COL], k, "+dropla")
                X_fit = X_train[y == 0]
                extra = f" n_dropped={int(y.sum())}"
            if not M.is_supervised(m):
                s = M.scores(m, X_fit, X_test, seed)
                if n := M.POSINF.get(m):  # TabPFN 비지도: +inf (밀도 0) 를 맨 위 유한 점수로 바꾼 행 수 (결정 30)
                    extra += f" posinf={n}"
            out = OUTPUTS / comp.name / m
            out.mkdir(parents=True, exist_ok=True)
            clear_test_info(out)
            np.save(out / "test.npy", s)
            run = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
            write_test_info(out, run, float("nan"), 0, None, M.opts(m) + extra)
            rows.append(dict(timestamp=run, comp=comp.name, model=m, metric="auroc", cv=None, n_folds=0,
                             seconds=round(time.time() - t0, 1), note=args.note, test=True, opts=M.opts(m) + extra,
                             n_features=len(feats)))
            print(f"[anomaly] {comp.name} {m}: {time.time() - t0:.1f}s (context {len(X_fit)}행, test {len(X_test)}행)",
                  flush=True)
        except Exception:  # noqa: BLE001 - 방법 하나가 실패해도 나머지는 돌림
            failed.append(m)
            print(f"[anomaly] {comp.name} {m} 실패:\n{traceback.format_exc()}", flush=True)
    if rows:
        append_rows(pd.DataFrame(rows))
    if failed:
        print(f"[anomaly] 실패한 방법: {', '.join(failed)}")
        return 1
    return 0
