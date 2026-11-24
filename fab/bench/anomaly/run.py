"""원격 전용: 이상탐지 대회 하나에서 방법들을 차례로 → <대회>/<방법>/test.npy (test.csv 행 순서의 점수, 클수록 이상) + test.json,
results.csv 에 행. fab/runner/run.py 의 main 이 anomaly 대회를 여기로 넘김 (fab databricks 실행기가 `fab run` 을 부르므로).
출력 자리 · 꼴이 FM 실행과 같아 정리 작업의 fm 단계가 그대로 합침 (fab/remote/reconcile.py).

    fab run <대회> <방법...> --note <글> [--train-labels <Volume 경로>] [--tabpfn-secret <scope>/<key>]

--train-labels: +clean 방법 (깨끗한 context 진단) 이 읽는 train 라벨 폴더 (내 Volume, remote.py).
--tabpfn-secret: TabPFN v3.5 가중치의 라이선스 동의 토큰이 든 Databricks secret. 토큰 값은 이 원격 작업 안에서만 읽고
(databricks-sdk, 작업 신원 = 내 신원) 환경 변수 TABPFN_TOKEN 으로만 넘김. 출력 · 결과에 남기지 않음.
"""

from __future__ import annotations

import argparse
import base64
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


def tabpfn_token(secret: str | None):
    """TABPFN_TOKEN 이 없으면 Databricks secret 에서 (값은 출력하지 않음)."""
    if os.environ.get("TABPFN_TOKEN"):
        return
    if not secret or "/" not in secret:
        raise SystemExit("TabPFN v3.5: TABPFN_TOKEN 도 --tabpfn-secret <scope>/<key> 도 없음")
    from databricks.sdk import WorkspaceClient

    scope, key = secret.split("/", 1)
    value = WorkspaceClient().secrets.get_secret(scope=scope, key=key).value
    os.environ["TABPFN_TOKEN"] = base64.b64decode(value).decode().strip()


def main(argv: list[str]):
    ap = argparse.ArgumentParser(prog="fab run (anomaly)")
    ap.add_argument("comp")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--note", default="")
    ap.add_argument("--train-labels", default=None)
    ap.add_argument("--tabpfn-secret", default=None)
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
    rows, failed = [], []
    for m in args.methods:
        t0 = time.time()
        try:
            X_fit = X_train
            if M.is_clean(m):  # 깨끗한 context 진단: train 의 정상 행만 (train 라벨을 씀 → Elo 밖)
                if not args.train_labels:
                    raise RuntimeError("+clean 은 --train-labels 가 필요")
                lab = pd.read_csv(Path(args.train_labels) / f"{comp.name}.csv").set_index(ID_COL)[LABEL_COL]
                X_fit = X_train[lab.loc[train[ID_COL]].to_numpy() == 0]
            if M.is_gpu(m):
                tabpfn_token(args.tabpfn_secret)
            s = M.scores(m, X_fit, X_test, seed)
            out = OUTPUTS / comp.name / m
            out.mkdir(parents=True, exist_ok=True)
            clear_test_info(out)
            np.save(out / "test.npy", s)
            run = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
            write_test_info(out, run, float("nan"), 0, None, M.opts(m))
            rows.append(dict(timestamp=run, comp=comp.name, model=m, metric="auroc", cv=None, n_folds=0,
                             seconds=round(time.time() - t0, 1), note=args.note, test=True, opts=M.opts(m),
                             n_features=len(feats)))
            print(f"[anomaly] {comp.name} {m}: {time.time() - t0:.1f}s (context {len(X_fit)}행, test {len(X_test)}행)",
                  flush=True)
        except Exception:  # noqa: BLE001 - 방법 하나가 실패해도 나머지는 돌림
            failed.append(m)
            print(f"[anomaly] {comp.name} {m} 실패:\n{traceback.format_exc()}", flush=True)
    # tabpfn 이 동의 확인 뒤 토큰을 ~/.cache/tabpfn/auth_token 에 저장함 → 지움 (실행기가 모델 캐시를 Volume 에 묶을 때
    # XDG_CACHE_HOME/tabpfn 만 묶어 여기는 안 들어가지만, 같은 자리가 되는 환경도 있을 수 있어서 둘 다)
    for d in (Path.home() / ".cache" / "tabpfn", Path(os.environ.get("XDG_CACHE_HOME") or "/nonexistent") / "tabpfn"):
        (d / "auth_token").unlink(missing_ok=True)
    if rows:
        append_rows(pd.DataFrame(rows))
    if failed:
        print(f"[anomaly] 실패한 방법: {', '.join(failed)}")
        return 1
    return 0
