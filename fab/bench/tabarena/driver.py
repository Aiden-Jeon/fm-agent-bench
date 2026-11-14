"""TabArena 옆 가상환경 (FAB_TABARENA/.venv: openml·pyarrow·pandas) 에서 도는 보조 스크립트. fab 을 import 하지 않음.

로컬: fab/bench/tabarena/__init__.py 가 `python -I` 로 부름 (cwd 는 받은 데이터 밖).

  fetch <task_id> <out>        OpenML task 의 데이터·고정 분할 → <out>/data.csv · splits.csv · meta.json · openml.json
  reference <url> <out_csv>    TabArena 공개 결과 parquet → CSV (기본 설정 행 `_c1_` 만)

fetch 의 결과 (분할을 다시 나누지 않음, OpenML 의 Task_<id>_splits.arff 그대로):
  data.csv    OpenML 데이터 그대로의 행 순서 (위치 = OpenML 행 번호). 피처 + 타깃 열
  splits.csv  (repeat, fold, idx): 그 split 의 TEST 행 위치. TRAIN 은 같은 repeat 안의 나머지 (fetch 가 확인)
  meta.json   타깃·범주형 열·클래스·split 수
  openml.json OpenML 데이터셋 설명 (description.md 를 쓸 때 참고만, 에이전트에게는 안 줌)
결과 JSON 은 stdout 마지막 줄.
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (fab-bench)"}  # data.tabarena.ai 는 urllib 기본 UA 를 403 으로 막음


def fetch(task_id: str, out: str) -> dict:
    import numpy as np
    import openml

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(out.parent / "_openml"))  # ~/.openml 대신 받은 데이터 옆
    task = openml.tasks.get_task(int(task_id), download_splits=True, download_data=True,
                                 download_qualities=False, download_features_meta_data=False)
    ds = task.get_dataset()
    X, y, cat, names = ds.get_data(target=task.target_name)
    n_repeats, n_folds, _ = task.get_split_dimensions()
    rows = []
    for r in range(n_repeats):
        seen = np.zeros(len(X), dtype=int)
        for f in range(n_folds):
            tr, te = task.get_train_test_split_indices(repeat=r, fold=f, sample=0)
            if len(set(tr) & set(te)) or len(tr) + len(te) != len(X):
                raise SystemExit(f"task {task_id} r{r}f{f}: TRAIN·TEST 가 전체를 나누지 않음")
            seen[te] += 1
            rows += [(r, f, int(i)) for i in sorted(te)]
        if not (seen == 1).all():
            raise SystemExit(f"task {task_id} repeat {r}: 모든 행이 TEST 에 한 번씩이 아님")
    df = X.copy()
    df[task.target_name] = y
    df.to_csv(out / "data.csv", index=False)
    with open(out / "splits.csv", "w") as fh:
        fh.write("repeat,fold,idx\n")
        fh.writelines(f"{r},{f},{i}\n" for r, f, i in rows)
    is_clf = "Classification" in str(task.task_type)
    meta = dict(task_id=int(task_id), dataset_id=int(ds.dataset_id), dataset=ds.name, target=task.target_name,
                features=list(names), categorical=[n for n, c in zip(names, cat) if c],
                task_type="classification" if is_clf else "regression",
                class_labels=[str(c) for c in task.class_labels] if is_clf else None,
                n_rows=int(len(X)), n_repeats=int(n_repeats), n_folds=int(n_folds),
                y_dtype=str(y.dtype), x_dtypes={c: str(t) for c, t in X.dtypes.items()})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "openml.json").write_text(json.dumps(dict(name=ds.name, description=ds.description,
                                                     original_data_url=ds.original_data_url,
                                                     citation=ds.citation, licence=ds.licence), indent=1))
    return dict(out=str(out), n_rows=meta["n_rows"], n_repeats=meta["n_repeats"], n_folds=meta["n_folds"])


def reference(url: str, out_csv: str) -> dict:
    import pandas as pd

    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        df = pd.read_parquet(io.BytesIO(r.read()))
    df = df[df["method"].str.contains("_c1_")]  # 기본 설정 (r* = 무작위 설정, 튜닝 결과는 따로 안 받음)
    keep = [c for c in ["dataset", "fold", "method", "metric", "problem_type", "metric_error", "metric_error_val",
                        "time_train_s", "time_infer_s", "imputed", "ta_name", "ta_suite"] if c in df]
    df[keep].to_csv(out_csv, index=False)
    return dict(rows=len(df), methods=sorted(df["method"].unique()))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    print(json.dumps({"fetch": fetch, "reference": reference}[cmd](*args), default=str))
