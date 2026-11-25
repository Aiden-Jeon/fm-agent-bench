"""fev 옆 가상환경 (FAB_FORECAST/.venv: fev 0.10.0 · datasets) 에서 도는 보조 스크립트. fab 을 import 하지 않음.

로컬: fab/bench/forecast/__init__.py 가 `python -I` 로 부름 (cwd 는 받은 데이터 밖). HF 캐시는 HF_HOME (FAB_FORECAST/hf).

  window <task.json> <window> <out>      fev Task 의 window 하나 → <out>/past.csv · future.csv · answers.csv · static.csv · meta.json
  score <task.json> <window> <pred.csv>  예측 (id·timestamp·predictions·분위수 열, 원래 id) 을 fev 의 compute_metrics 로 채점 (확인용)

window 의 결과 (fev 의 get_input_data · get_ground_truth 그대로, as_univariate. 단변량 task 만):
  past.csv     id · timestamp · target · 동적 공변량 (past · known). cutoff 앞 전부, id 순 · 시각 순 (fev 의 순서)
  future.csv   id · timestamp · known 공변량 (맞힐 칸)
  answers.csv  id · timestamp · target (정답, 에이전트에게 안 줌)
  static.csv   id · static 공변량 (있을 때만)
  meta.json    task 정의 · cutoff · freq · 시리즈 수 · 열 이름 · 원래 타깃 이름
결과 JSON 은 stdout 마지막 줄.
"""

import json
import sys
from pathlib import Path


def _task(task_json: str):
    import fev

    return fev.Task(**json.loads(Path(task_json).read_text()))


def window(task_json: str, idx: str, out: str) -> dict:
    import fev

    task = _task(task_json)
    if task.is_multivariate:
        raise SystemExit(f"{task.task_name}: 다변량 task 는 아직 안 받음 (docs/forecast.md 2 의 규칙 1)")
    w = task.get_window(int(idx))
    past, future = w.get_input_data()
    gt = w.get_ground_truth()
    past_df, future_df, static_df = fev.convert_input_data(w, adapter="pandas", as_univariate=False)
    gt_df = fev.adapters.PandasAdapter._to_long_df(gt, id_column=task.id_column)
    y = task.target_columns[0]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ren = {task.id_column: "id", task.timestamp_column: "timestamp", y: "target"}
    dyn = [*task.past_dynamic_columns, *task.known_dynamic_columns]
    past_df = past_df.rename(columns=ren)[["id", "timestamp", "target", *dyn]]
    future_df = future_df.rename(columns=ren)[["id", "timestamp", *task.known_dynamic_columns]]
    gt_df = gt_df.rename(columns=ren)[["id", "timestamp", "target"]]
    if (future_df[["id", "timestamp"]].astype(str).to_numpy() != gt_df[["id", "timestamp"]].astype(str).to_numpy()).any():
        raise SystemExit("future 와 정답의 (id, timestamp) 가 다름")
    past_df.to_csv(out / "past.csv", index=False)
    future_df.to_csv(out / "future.csv", index=False)
    gt_df.to_csv(out / "answers.csv", index=False)
    if static_df is not None:
        static_df.rename(columns={task.id_column: "id"}).to_csv(out / "static.csv", index=False)
    meta = dict(task=task.to_dict(), task_name=task.task_name, window=int(idx), num_windows=task.num_windows,
                cutoff=str(w.cutoff), cutoffs=[str(c) for c in task.cutoffs], freq=task.freq, horizon=task.horizon,
                seasonality=task.seasonality, quantile_levels=task.quantile_levels, target_from=y,
                n_items=len(past), n_items_full=len(task.load_full_dataset()), n_past=len(past_df),
                past_dynamic=task.past_dynamic_columns, known_dynamic=task.known_dynamic_columns,
                static=task.static_columns, dtypes={c: str(t) for c, t in past_df.dtypes.items()},
                fingerprint=task._dataset_fingerprint, fev_version=fev.__version__)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return dict(out=str(out), n_items=meta["n_items"], n_past=meta["n_past"], cutoff=meta["cutoff"])


def score(task_json: str, idx: str, pred_csv: str) -> dict:
    """확인용: 우리 채점 (fab/bench/forecast/metrics.py) 이 fev 와 같은 값인지. 예측은 정답과 같은 (id, timestamp) 순서."""
    import datasets
    import pandas as pd

    task = _task(task_json)
    w = task.get_window(int(idx))
    pred = pd.read_csv(pred_csv)
    qs = [str(q) for q in task.quantile_levels]
    rows = []
    for _, g in pred.groupby("id", sort=False):
        rows.append({"predictions": g["predictions"].astype(float).tolist(), **{q: g[q].astype(float).tolist() for q in qs}})
    preds = datasets.DatasetDict({task.target_columns[0]: datasets.Dataset.from_list(rows)})
    from fev.metrics import get_metric

    metrics = [get_metric(m) for m in [task.eval_metric, *task.extra_metrics]]
    return w.compute_metrics(preds, metrics, task.seasonality, task.quantile_levels)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    print(json.dumps({"window": window, "score": score}[cmd](*args), default=str))
