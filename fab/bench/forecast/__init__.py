"""fev-bench 시계열 예측 트랙: task 의 window 하나를 대회로 준비 (로컬), 채점 (원격 정리 작업, remote.py · fab/remote),
fev-bench 공개 결과 · Elo. 설계 · 결정은 docs/forecast.md.

    uv run --frozen fab fc prepare redset_1H                 # 마지막 window → competitions/fc-redset-1h/data/ (public), 정답은 FAB_FORECAST
    uv run --frozen fab fc describe redset_1H                # 에이전트용 description.md (describe.py, 덮어씀)
    uv run --frozen fab fc fm all [--methods tabicl_ts]      # FM · 기존 방법 원격 (모든 window) → fab remote kick → results/forecast.csv
    uv run --frozen fab fc check redset_1H                   # 로컬 확인: 우리 채점 = fev 채점, 형식 검사, 누출 검사
    uv run --frozen fab fc reference                         # fev-bench 공개 결과 (10 task) → results/forecast_reference.csv
    uv run --frozen fab fc elo                               # Elo (Seasonal Naive = 1000) → results/forecast_elo.csv
    uv run --frozen fab fc agent run fc-redset-1h aide-sonnet  # 에이전트 (마지막 window, 익명 id)

대회 단위: (fev task, window) 하나. 저장소에는 마지막 window (Agent 용, 결정 2) 만 competitions/fc-<task>/ (config.yaml ·
description.md). FM · 기존 방법은 원격 작업이 fev 로 모든 window 를 직접 돌림 (run_fev.py) → 대회 폴더 없음.

데이터: fev 0.10.0 · datasets 가 이 프로젝트 uv 환경에 없음 (uv.lock 을 안 바꿈) → 옆 가상환경에서 driver.py 를 `-I` 로:

    uv venv --python 3.12 $FAB_FORECAST/.venv
    uv pip install --python $FAB_FORECAST/.venv/bin/python fev==0.10.0

FAB_FORECAST (절대경로, 기본 ../forecast) 아래: .venv/, hf/ (HF 캐시), data/cache/<task>/w<window>/ (driver 결과, 원래 id),
data/answers/<대회>.{csv,json} (정답 · 계절 오차 · id 대응표). 대회 data/ 에는 public 만 (train.csv · test.csv ·
sample_submission.csv · description.md). 시리즈 id 는 새로 매김 (s001…, 무작위 순서), 타깃 열은 target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fab.core.competition import COMP_DIR, ROOT, Competition, load_competition
from fab.core.results import RESULTS

TASKS = Path(__file__).with_name("tasks.yaml")  # fev-bench task 정의 중 10개 (fev @ ecce671, Apache-2.0)
FC_DIR = Path(os.environ.get("FAB_FORECAST", ROOT.parent / "forecast")).expanduser().resolve()
CACHE = FC_DIR / "data" / "cache"
ANSWERS = FC_DIR / "data" / "answers"
DRIVER = Path(__file__).with_name("driver.py")
PREFIX = "fc-"
ANON = "forecast-p"  # 익명 id: forecast-p01 … (tasks.yaml 의 순서)
ID_COL, TS_COL, POINT = "id", "timestamp", "predictions"
GRADES = RESULTS / "forecast.csv"  # 우리 실행 (FM · 기존 방법) 의 window 별 · task 평균 채점 (git 추적)
REFERENCE = RESULTS / "forecast_reference.csv"  # fev-bench 공개 결과 (git 추적, fab fc reference)
RUNS = RESULTS / "forecast_runs.csv"  # 에이전트 실행별 채점 (git 추적, 원격 정리 작업이 씀)
ELO = RESULTS / "forecast_elo.csv"
ANCHOR = "Seasonal Naive"  # Elo 기준점 (결정 4)
FEV_COMMIT = "ecce67120aae4827e9702b978a9e9327ecde81d0"
FEV_RESULTS = f"https://raw.githubusercontent.com/autogluon/fev/{FEV_COMMIT}/benchmarks/fev_bench/results"
# 공개 결과에서 받는 방법 (파일 이름 → 풀의 이름). AutoARIMA 는 공개 결과로 FM 풀에만 (결정 5)
REFERENCE_METHODS = {"seasonal_naive": "Seasonal Naive", "autoets": "AutoETS", "autoarima": "AutoARIMA",
                     "chronos-2": "Chronos-2", "chronos-bolt": "Chronos-Bolt", "timesfm-2_5": "TimesFM-2.5",
                     "tabpfn-ts": "TabPFN-TS (v2)", "tabpfn-ts-3": "TabPFN-TS-3"}
REFERENCE_COLS = ["model_name", "task_name", "test_error", "SQL", "MASE", "WQL", "WAPE", "num_windows", "num_forecasts",
                  "training_time_s", "inference_time_s", "trained_on_this_dataset", "fev_version", "fev_commit",
                  "model_kwargs", "dataset_fingerprint"]


# --- task · 대회 ------------------------------------------------------------------------------------------------


@cache
def tasks() -> dict[str, dict]:
    """fev task 이름 → 정의 (tasks.yaml 순서)."""
    return {(t.get("task_name") or t["dataset_config"]): t for t in yaml.safe_load(TASKS.read_text())["tasks"]}


def task_def(name: str) -> dict:
    if name not in tasks():
        raise SystemExit(f"이 트랙의 task 가 아님: {name} (fab/bench/forecast/tasks.yaml: {', '.join(tasks())})")
    return tasks()[name]


def num_windows(name: str) -> int:
    return int(task_def(name).get("num_windows", 1))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def comp_name(name: str) -> str:
    return PREFIX + slug(name)


def anon_id(name: str) -> str:
    return f"{ANON}{list(tasks()).index(name) + 1:02d}"


def committed() -> dict[str, Competition]:
    """저장소의 대회 (competitions/fc-*) {fev task: 대회}."""
    out = {}
    for p in sorted(COMP_DIR.glob(f"{PREFIX}*/config.yaml")):
        c = load_competition(p.parent.name)
        if c.forecast:
            out[c.forecast["task"]] = c
    return out


def resolve(target: str) -> str:
    """fev task 이름 또는 대회 이름 → fev task 이름."""
    if (COMP_DIR / target / "config.yaml").exists() and (fc := load_competition(target).forecast):
        return fc["task"]
    task_def(target)
    return target


def default_config(name: str) -> dict:
    """대회 설정 (마지막 window). competitions/fc-<task>/config.yaml 를 처음 만들 때."""
    w = num_windows(name) - 1
    return dict(name=comp_name(name), forecast=dict(task=name, window=w, anon_id=anon_id(name)), task="forecast",
                target="target", id_col=ID_COL, metric="sql", cv={})


# --- 받기 (driver) -----------------------------------------------------------------------------------------------


def _python() -> Path:
    py = FC_DIR / ".venv" / "bin" / "python"
    if not py.exists():
        sys.exit(f"forecast 가상환경 없음: {py}\n설치 방법은 fab/bench/forecast/__init__.py 위쪽 설명 (FAB_FORECAST 로 위치 지정)")
    return py


def _driver(*args: str) -> dict:
    # -I: 받은 파일이 있는 곳의 모듈을 import 하지 않게. cwd 도 받은 데이터 밖 (임시 폴더)
    env = dict(os.environ, HF_HOME=str(FC_DIR / "hf"))
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(_python()), "-I", str(DRIVER), *args], stdout=subprocess.PIPE, text=True, check=True,
                             cwd=d, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def fetched(name: str, window: int) -> Path:
    """fev 의 window 하나 (원래 id, 한 번만) → CACHE/<task>/w<window>/."""
    d = CACHE / slug(name) / f"w{window}"
    if not (d / "meta.json").exists():
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(task_def(name), f)
        try:
            print(f"fev {name} window {window} → {d}")
            _driver("window", f.name, str(window), str(d))
        finally:
            os.unlink(f.name)
    return d


# --- 대회 만들기 -------------------------------------------------------------------------------------------------


def id_map(name: str, ids: list[str]) -> dict[str, str]:
    """원래 시리즈 id → 익명 id (s001…). 순서는 task 이름에서 정한 seed 의 무작위 (원래 순서 · 이름을 주지 않음)."""
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(len(ids))
    width = max(3, len(str(len(ids))))
    return {ids[i]: f"s{k + 1:0{width}d}" for k, i in enumerate(order)}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={ID_COL: str}, keep_default_na=False, na_values=[""])


def prepare(comp: Competition):
    """fab download · fab fc prepare 가 부름: window 의 public → comp.data_dir, 정답 → ANSWERS/<대회>.{csv,json}."""
    fc = comp.forecast
    name, w = fc["task"], int(fc["window"])
    d = fetched(name, w)
    meta = json.loads((d / "meta.json").read_text())
    past, future, ans = _read(d / "past.csv"), _read(d / "future.csv"), _read(d / "answers.csv")
    ids = list(dict.fromkeys(past[ID_COL]))
    m = id_map(name, ids)
    for df in (past, future, ans):
        df[ID_COL] = df[ID_COL].map(m)
    if (d / "static.csv").exists():
        static = _read(d / "static.csv")
        static[ID_COL] = static[ID_COL].map(m)
        past = past.merge(static, on=ID_COL, how="left")
    # 시리즈는 익명 id 순, 시리즈 안은 시각 순 (fev 의 순서). 정답 · test 도 같은 순서
    key = lambda df: df.sort_values([ID_COL, TS_COL], kind="stable").reset_index(drop=True)  # noqa: E731
    past, future, ans = key(past), key(future), key(ans)
    h = meta["horizon"]
    if (ans.groupby(ID_COL).size() != h).any():
        raise SystemExit(f"{comp.name}: 정답이 시리즈마다 horizon {h} 개가 아님")
    # 계절 오차 (채점용, metrics.seasonal_error) 는 public train 의 target 에서 → 정답 옆에 둠 (채점 때 train 을 안 읽게)
    from fab.bench.forecast.metrics import seasonal_error

    groups = {k: g.target.to_numpy(float) for k, g in past.groupby(ID_COL, sort=True)}
    scale = dict(zip(groups, seasonal_error(list(groups.values()), meta["seasonality"])))
    levels = meta["quantile_levels"]
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    past.to_csv(comp.data_dir / "train.csv", index=False)
    future.to_csv(comp.data_dir / "test.csv", index=False)
    last = past.groupby(ID_COL).target.last()
    sample = future[[ID_COL, TS_COL]].assign(**{POINT: future[ID_COL].map(last).fillna(0.0)})
    for q in levels:
        sample[str(q)] = sample[POINT]
    sample.to_csv(comp.data_dir / "sample_submission.csv", index=False)
    if (comp.dir / "description.md").exists():
        shutil.copy(comp.dir / "description.md", comp.data_dir / "description.md")
    ANSWERS.mkdir(parents=True, exist_ok=True)
    ans.assign(scale=ans[ID_COL].map(scale)).to_csv(ANSWERS / f"{comp.name}.csv", index=False)
    info = dict(comp=comp.name, task=name, window=w, num_windows=meta["num_windows"], cutoff=meta["cutoff"],
                horizon=h, seasonality=meta["seasonality"], quantile_levels=levels, freq=meta["freq"],
                n_items=len(ids), n_train=len(past), n_test=len(future), target_from=meta["target_from"],
                fingerprint=meta["fingerprint"], fev_version=meta["fev_version"], ids=m)
    (ANSWERS / f"{comp.name}.json").write_text(json.dumps(info, indent=1))
    print(f"{comp.name}: 시리즈 {len(ids)} · train {len(past)} · test {len(future)} → {comp.data_dir} "
          f"(정답 {ANSWERS / comp.name}.csv)")


def ensure_prepared(comp: Competition):
    """없으면 prepare. 있으면 description.md 만 대회 폴더의 것으로 맞춤 (TabArena 와 같음)."""
    if not ((comp.data_dir / "train.csv").exists() and (ANSWERS / f"{comp.name}.json").exists()):
        return prepare(comp)
    src, dst = comp.dir / "description.md", comp.data_dir / "description.md"
    if src.exists() and (not dst.exists() or dst.read_bytes() != src.read_bytes()):
        shutil.copy(src, dst)
        print(f"{comp.name}: description.md 갱신")


def make_comp(name: str) -> Competition:
    """competitions/fc-<task>/config.yaml 이 없으면 만들고 (description.md 는 손으로 씀) 준비."""
    d = COMP_DIR / comp_name(name)
    if not (d / "config.yaml").exists():
        d.mkdir(parents=True, exist_ok=True)
        cfg = default_config(name)
        head = (f"# fev-bench task {name} 의 마지막 window ({cfg['forecast']['window']} / {num_windows(name)} 의 0 부터 센 번호). "
                f"anon_id = 에이전트에게 보이는 이름 (docs/forecast.md 8)\n")
        (d / "config.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    comp = load_competition(d.name)
    ensure_prepared(comp)
    return comp


# --- 채점 -------------------------------------------------------------------------------------------------------


def read_answers(path: Path) -> pd.DataFrame:
    return _read(path)


def score_frame(sub: pd.DataFrame, ans: pd.DataFrame, info: dict) -> dict:
    """제출 표 (id · timestamp · predictions · 분위수 열) → {valid_submission, score (= SQL), metric_error (= SQL), MASE,
    WQL, WAPE} 또는 {valid_submission: False, error}. 에이전트 · FM 이 같이 씀. sub 는 고치지 않음."""
    from fab.bench.forecast.metrics import scores

    levels = info["quantile_levels"]
    want = [ID_COL, TS_COL, POINT, *[str(q) for q in levels]]
    if list(map(str, sub.columns)) != want:
        return dict(valid_submission=False, error=f"열 {list(sub.columns)} ≠ {want}")
    sub = sub.copy()
    sub.columns = want
    # timestamp 는 글자가 아니라 시각으로 비교 ('2024-01-01' 과 '2024-01-01 00:00:00' 은 같은 칸)
    sub[ID_COL] = sub[ID_COL].astype(str)
    sub[TS_COL] = pd.to_datetime(sub[TS_COL].astype(str).str.strip(), errors="coerce", format="mixed")
    if sub[TS_COL].isna().any():
        return dict(valid_submission=False, error=f"timestamp 를 시각으로 못 읽는 행 {int(sub[TS_COL].isna().sum())}개")
    if len(sub) != len(ans):
        return dict(valid_submission=False, error=f"행 수 {len(sub)} ≠ test {len(ans)}")
    sub = sub.drop_duplicates([ID_COL, TS_COL])
    keys = pd.MultiIndex.from_arrays([ans[ID_COL].astype(str), pd.to_datetime(ans[TS_COL], format="mixed")])
    sub = sub.set_index([ID_COL, TS_COL])
    if len(sub) != len(ans) or not keys.isin(sub.index).all():
        return dict(valid_submission=False, error="(id, timestamp) 가 test 와 다름 (빠지거나 겹침)")
    sub = sub.loc[keys]
    try:
        pred = sub.to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        return dict(valid_submission=False, error=f"예측이 숫자가 아님: {e!r}"[:300])
    if not np.isfinite(pred).all():
        return dict(valid_submission=False, error="예측에 NaN/inf")
    n, h = ans[ID_COL].nunique(), info["horizon"]
    y = ans["target"].to_numpy(float).reshape(n, h)
    s = ans.groupby(ID_COL, sort=False).scale.first().to_numpy(float)
    r = scores(y, pred[:, 0].reshape(n, h), pred[:, 1:].reshape(n, h, len(levels)), levels, s)
    return dict(valid_submission=True, score=r["SQL"], metric_error=r["SQL"], **r)


def score_submission(sub_csv: Path, answers_csv: Path, info: dict) -> dict:
    ans = read_answers(answers_csv)
    try:
        sub = pd.read_csv(sub_csv, dtype={ID_COL: str, TS_COL: str})
    except Exception as e:  # noqa: BLE001
        return dict(valid_submission=False, error=f"읽기 실패: {e!r}")
    return score_frame(sub, ans, info)


# --- 로컬 확인 ---------------------------------------------------------------------------------------------------


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, m: int, levels: list[float]) -> pd.DataFrame:
    """확인용 예측: 시리즈마다 마지막 계절을 되풀이 (점 · 분위수 같음. statsforecast 의 구간과는 다름)."""
    out = []
    for k, g in test.groupby(ID_COL, sort=False):
        y = train.loc[train[ID_COL] == k, "target"].to_numpy(float)
        last = y[-m:] if len(y) >= m else np.full(m, y[-1] if len(y) else 0.0)
        last = np.where(np.isnan(last), np.nanmean(y) if np.isfinite(y).any() else 0.0, last)
        p = np.resize(last, len(g))
        out.append(g[[ID_COL, TS_COL]].assign(**{POINT: p}, **{str(q): p * (0.8 + 0.4 * q) for q in levels}))
    return pd.concat(out, ignore_index=True)


def check(name: str) -> dict:
    """원격 전에: 우리 채점 = fev 채점 (확인용 예측 두 개), 정답 = fev 의 정답, sample · 확인용 예측이 형식 검사를 통과,
    엉터리 제출은 무효, 누출 검사. 결과 dict (fab fc check 가 출력)."""
    comp = make_comp(name)
    info = json.loads((ANSWERS / f"{comp.name}.json").read_text())
    ans = read_answers(ANSWERS / f"{comp.name}.csv")
    train, test = _read(comp.data_dir / "train.csv"), _read(comp.data_dir / "test.csv")
    sample = pd.read_csv(comp.data_dir / "sample_submission.csv", dtype={ID_COL: str, TS_COL: str})
    inv = {v: k for k, v in info["ids"].items()}
    out = {}
    with tempfile.TemporaryDirectory() as d, tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(task_def(name), tf)
        tf.close()
        for label, pred in [("sample", sample), ("snaive", seasonal_naive(train, test, info["seasonality"],
                                                                            info["quantile_levels"]))]:
            ours = score_frame(pred, ans, info)
            # fev 는 원래 id · fev 의 시리즈 순서 (원래 id 순) 로
            p = pred.assign(**{ID_COL: pred[ID_COL].map(inv)})
            order = {k: i for i, k in enumerate(dict.fromkeys(_read(fetched(name, info["window"]) / "answers.csv")[ID_COL]))}
            p = p.sort_values([ID_COL, TS_COL], key=lambda s: s.map(order) if s.name == ID_COL else s, kind="stable")
            p.to_csv(Path(d) / "p.csv", index=False)
            ref = _driver("score", tf.name, str(info["window"]), str(Path(d) / "p.csv"))
            diff = {k: abs(ours[k] - ref[k]) / max(abs(ref[k]), 1e-12) for k in ("SQL", "MASE", "WQL", "WAPE")}
            out[label] = dict(ours={k: round(ours[k], 6) for k in diff}, fev={k: round(ref[k], 6) for k in diff},
                              max_rel_diff=max(diff.values()))
        os.unlink(tf.name)
    raw = _read(fetched(name, info["window"]) / "answers.csv")
    raw[ID_COL] = raw[ID_COL].map(info["ids"])
    merged = ans.merge(raw, on=[ID_COL, TS_COL], suffixes=("", "_fev"))
    out["answers_match_fev"] = bool(len(merged) == len(ans) and
                                    np.allclose(merged.target, merged.target_fev, equal_nan=True))
    bad = {"열 빠짐": sample.drop(columns=["0.5"]), "행 빠짐": sample.iloc[1:], "NaN": sample.assign(**{POINT: np.nan}),
           "timestamp 틀림": sample.assign(**{TS_COL: "1999-01-01"})}
    out["invalid_rejected"] = {k: not score_frame(v, ans, info)["valid_submission"] for k, v in bad.items()}
    from fab.agent import find_leaks

    texts = {f: (comp.data_dir / f).read_text() for f in ("description.md",) if (comp.data_dir / f).exists()}
    texts["ids"] = " ".join(train[ID_COL].unique())
    texts["columns"] = " ".join(train.columns) + " " + " ".join(test.columns)
    out["leaks"] = find_leaks(leak_terms(comp, info), texts)
    return out


def leak_terms(comp: Competition, info: dict | None = None) -> list[str]:
    """에이전트에게 보이면 안 되는 글: fev task · 데이터셋 config 이름 (구분자 바꾼 꼴), 대회 이름, 원래 시리즈 id (짧은 숫자 id 는 뺌)."""
    info = info or json.loads((ANSWERS / f"{comp.name}.json").read_text())
    name = comp.forecast["task"]
    base = task_def(name).get("dataset_config", name)
    words = set()
    for n in {name.lower(), base.lower()}:  # 전체 이름만 (첫 낱말 'solar'·'hospital' 은 영역을 뜻하는 일반어라 설명에 씀)
        words |= {n, n.replace("_", "-"), n.replace("_", " "), n.replace("/", "_")}
    words.add(comp.name)
    words |= {k for k in info["ids"] if not k.isdigit() and len(k) >= 4}
    return sorted(w for w in words if len(w) >= 3)


# --- FM · 기존 방법 (원격, run_fev.py) ---------------------------------------------------------------------------

RUN_FEV = Path(__file__).with_name("run_fev.py")
FEV_VERSION = "0.10.0"
# 방법 키 (run_fev.METHODS) → 결과표의 이름. 공개 결과 (REFERENCE_METHODS) 와 같은 이름이면 FM 풀에서 "(ours)" 를 붙임
METHOD_NAMES = {"snaive": "Seasonal Naive", "autoets": "AutoETS", "chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt",
                "timesfm25": "TimesFM-2.5", "tabpfn_ts": "TabPFN-TS (v2)", "tabicl_ts": "TabICL-TS"}
# 기본 작업 묶음 (GPU 작업 하나씩 동시에): 오래 걸리는 FM 둘은 따로, 나머지는 한 작업 (docs/forecast.md 10)
DEFAULT_GROUPS = [["tabpfn_ts"], ["tabicl_ts"], ["snaive", "autoets", "chronos2", "chronos_bolt", "timesfm25"]]


def submit_fm(methods: list[str], names: list[str], accelerator: str | None, timeout: int, note: str = "",
              dry_run: bool = False) -> dict:
    """(방법 묶음 하나) × task 들을 GPU 작업 하나로 (databricks air run, 내 신원). 제출 기록은 Volume 의 state/forecast/."""
    import yaml as _yaml

    from fab.runner.databricks import databricks, run_key
    from fab.runner.databricks import settings as dbx_settings

    if bad := [m for m in methods if m not in METHOD_NAMES]:
        raise SystemExit(f"알 수 없는 방법: {bad} (있는 것: {', '.join(METHOD_NAMES)})")
    cfg = dbx_settings(argparse.Namespace(accelerator=accelerator, image=None))
    n_gpus = re.match(r"GPU_(\d+)x", cfg["accelerator"])
    key = run_key("fc-" + "-".join(m.replace("_", "") for m in methods), "")
    out = f"{cfg['volume']}/runs/{key}"
    job = dict(methods=methods, tasks={n: task_def(n) for n in names}, out=out, fev_version=FEV_VERSION,
               fev_commit=FEV_COMMIT, task_timeout=min(timeout * 60, 6 * 3600))
    work = Path(tempfile.mkdtemp())
    try:
        code = work / "code"
        code.mkdir()
        shutil.copy(RUN_FEV, code / "run_fev.py")
        (code / "job.json").write_text(json.dumps(job, indent=1))
        (work / "air.yaml").write_text(_yaml.safe_dump({
            "experiment_name": "fab-forecast", "mlflow_run_name": key,
            "compute": {"accelerator_type": cfg["accelerator"], "num_accelerators": int(n_gpus[1])},
            "environment": {"version": 6}, "code_source": {"type": "snapshot", "snapshot": {"root_path": str(code)}},
            "command": "python $CODE_SOURCE_PATH/run_fev.py", "timeout_minutes": timeout, "max_retries": 0,
        }, sort_keys=False))
        if dry_run:
            return dict(key=key, job=job, air=(work / "air.yaml").read_text())
        res = databricks(cfg["profile"], "air", "run", "-f", str(work / "air.yaml"), "-o", "json", capture=True)
    finally:
        shutil.rmtree(work)
    info = json.loads(res.stdout[res.stdout.index("{"):])["data"]
    run = dict(key=key, run_id=info["run_id"], methods=methods, tasks=names, out=out, url=info["dashboard_url"],
               accelerator=cfg["accelerator"], note=note)
    from fab.remote import submit_item

    submit_item(cfg, "forecast", key, run)
    print(f"제출: {key} (run_id {run['run_id']}) {', '.join(methods)} × task {len(names)}개\n  {run['url']}")
    return run


def collect(m: dict, grader) -> tuple[list[dict], dict[str, str]]:
    """끝난 원격 작업 (submit_fm 의 기록) 의 결과 → results/forecast.csv 의 행. 원격 정리 작업이 부름 (내 신원, Volume).
    (task, 방법) 마다: window 별 fev 점수 (scorer=fev), task 평균 (window=all, fev evaluation_summary), 마지막 window 예측을
    Agent 와 같은 채점기로 (scorer=ours, is_last). 실패한 것은 errors 에 남기고 표에 안 씀."""
    from fab.core.results import append_rows

    now = pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds")
    rows, errors = [], {}
    res = Path(m["out"]) / "results"
    for name in m["tasks"]:
        comp = comp_name(name)
        info = grader.info(comp)
        for key in m["methods"]:
            d = res / slug(name) / key
            st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            if not st.get("ok"):
                errors[f"{name}/{key}"] = (st.get("error") or "결과 없음")[-500:]
                continue
            base = dict(timestamp=now, task=name, comp=comp, method=METHOD_NAMES[key], method_key=key, run=m["key"],
                        num_windows=st.get("num_windows"), fit_predict_s=st.get("fit_predict_s"),
                        versions=json.dumps(st.get("versions", {}), sort_keys=True),
                        relaxed_pins=" ".join(st.get("relaxed_pins") or []), note=m.get("note", ""))
            for w in json.loads((d / "windows.json").read_text()):
                rows.append(dict(base, window=f"w{w['window']}", cutoff=w["cutoff"], scorer="fev", is_last=False,
                                 **{k: w.get(k) for k in ("SQL", "MASE", "WQL", "WAPE")}))
            rows.append(dict(base, window="all", scorer="fev", is_last=False, **st.get("task_mean", {})))
            if info is None:
                errors[f"{name}/{key}"] = f"정답 없음 ({comp}): 마지막 window 를 채점 못 함 → fab fc prepare · 정답 업로드"
                continue
            sub = pd.read_csv(d / "last.csv", dtype={ID_COL: str, TS_COL: str})
            sub[ID_COL] = sub[ID_COL].map(info["ids"])
            r = grader.grade_frame(comp, sub)
            if not r.get("valid_submission"):
                errors[f"{name}/{key}"] = f"마지막 window 채점 실패: {r.get('error')}"
                continue
            rows.append(dict(base, window=f"w{info['window']}", scorer="ours", is_last=True,
                             **{k: r[k] for k in ("SQL", "MASE", "WQL", "WAPE")}))
    if rows:
        df = pd.DataFrame(rows)
        df["metric_error"] = df.SQL
        append_rows(df, GRADES)
    return rows, errors


# --- 공개 결과 · Elo ---------------------------------------------------------------------------------------------


def reference() -> pd.DataFrame:
    """fev-bench 공개 결과 (fev @ FEV_COMMIT 의 results/*.csv) 중 이 트랙의 task · 방법 → results/forecast_reference.csv."""
    import urllib.request

    parts = []
    for f, method in REFERENCE_METHODS.items():
        with urllib.request.urlopen(f"{FEV_RESULTS}/{f}.csv", timeout=120) as r:
            df = pd.read_csv(r)
        df = df[df.task_name.isin(tasks())]
        parts.append(df[[c for c in REFERENCE_COLS if c in df]].assign(method=method, source_file=f"{f}.csv"))
        print(f"{f}: {len(df)} / {len(tasks())} task")
    ref = pd.concat(parts, ignore_index=True).assign(snapshot=FEV_COMMIT[:7])
    ref.to_csv(REFERENCE, index=False)
    print(f"{REFERENCE}: {len(ref)}행")
    return ref


def elo_tables(rounds: int) -> pd.DataFrame:
    """FM 풀 (task 평균: 공개 결과 + 우리 실행) · Agent 풀 (마지막 window: 우리 실행 + Agent) × 지표 (SQL · MASE).
    Agent 풀은 우리 실행 · Agent 결과가 있어야 함 (없으면 FM 풀만)."""
    from fab.bench.tabarena import elo

    ref = pd.read_csv(REFERENCE)
    out = []
    for metric in ("SQL", "MASE"):
        pools = {"fm": ref[["method", "task_name", metric]].rename(columns={"task_name": "dataset", metric: "metric_error"})
                 .assign(split="all")}
        if GRADES.exists():
            g = pd.read_csv(GRADES)
            # FM 풀에는 공개 결과에 없는 우리 방법만 (TabICL-TS). 재현한 방법 (같은 이름) 을 같이 넣으면 거의 같은 값끼리
            # 부동소수 차로 승패가 갈려 숫자가 흔들림 (Seasonal Naive 가 같은 값인데 1023) → 재현은 맞춤 확인 (compare) 에만
            ours = g[(g.window == "all") & ~g.method.isin(set(ref.method))][["method", "task", metric]].rename(
                columns={"task": "dataset", metric: "metric_error"})
            pools["fm"] = pd.concat([pools["fm"], ours.assign(method=ours.method + " (ours)", split="all")])
            last = g[g.is_last.eq(True)][["method", "task", "window", metric]].rename(
                columns={"task": "dataset", "window": "split", metric: "metric_error"})
            agent = agent_long(metric)
            pools["agent"] = pd.concat([last, agent]) if len(agent) else last
        for pool, long in pools.items():
            long = long[long.dataset.isin(tasks())]
            filled = elo.fill(long, list(dict.fromkeys(long.method)), anchor=ANCHOR)
            t = elo.elo(filled, rounds=rounds, anchor=ANCHOR)
            out.append(t.reset_index().assign(pool=pool, metric=metric))
            with pd.option_context("display.width", 200):
                print(f"\n== {pool} · {metric}: task {int(t.datasets.iloc[0])}개")
                print(t.round({"elo": 0, "elo+": 0, "elo-": 0, "rank": 2, "winrate": 3, "imputed": 3}).to_string())
    res = pd.concat(out, ignore_index=True)
    cols = ["pool", "metric", "method", "elo", "elo+", "elo-", "rank", "winrate", "imputed", "datasets"]
    res[cols].round({"elo": 1, "elo+": 1, "elo-": 1, "rank": 3, "winrate": 4, "imputed": 4}).to_csv(ELO, index=False)
    print(f"\n{ELO}: {len(res)}행")
    return res


def agent_long(metric: str) -> pd.DataFrame:
    """Agent 실행 → method · dataset · split · metric_error · imputed (설정마다 회차 평균, 실패 회차는 NaN → fill 이
    Seasonal Naive 로 채우기 전에 회차 평균에서 빠지지 않게 여기서 Seasonal Naive 값을 넣음). +fm · +gw 는 뺌."""
    if not RUNS.exists():
        return pd.DataFrame(columns=["method", "dataset", "split", "metric_error", "imputed"])
    from fab.agent import FM_SUFFIX, GATEWAY_SUFFIX

    r = pd.read_csv(RUNS)
    r = r[~r.config.str.contains(FM_SUFFIX, regex=False) & ~r.config.str.contains(GATEWAY_SUFFIX, regex=False)]
    of = {c.name: (t, f"w{c.forecast['window']}") for t, c in committed().items()}
    r = r[r.comp.isin(of)].copy()
    r["dataset"], r["split"] = r.comp.map(lambda c: of[c][0]), r.comp.map(lambda c: of[c][1])
    r["failed"] = ~r.valid_submission.eq(True) | r[metric].isna()
    g = pd.read_csv(GRADES)
    sn = g[(g.method == ANCHOR) & g.is_last.eq(True)].set_index("task")[metric]
    r["metric_error"] = r[metric].where(~r.failed, r.dataset.map(sn))
    r["method"] = "agent: " + r.config
    return (r.groupby(["method", "dataset", "split"], as_index=False)
            .agg(metric_error=("metric_error", "mean"), imputed=("failed", "mean")))


# --- 명령 -------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    argv = sys.argv[2:] if argv is None else argv
    if argv[:1] == ["agent"]:
        from fab.agent import main as agent_main

        return agent_main(argv[1:], bench="forecast")
    ap = argparse.ArgumentParser(prog="fab fc", description="fev-bench 시계열 예측 (docs/forecast.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="마지막 window → 대회 (data/ 에 public, 정답은 FAB_FORECAST)")
    p.add_argument("tasks", nargs="+", help="fev task 또는 대회 이름, all = 10개")
    w = sub.add_parser("describe", help="competitions/fc-*/description.md 다시 만들기 (describe.py, 덮어씀)")
    w.add_argument("tasks", nargs="+")
    c = sub.add_parser("check", help="로컬 확인: 우리 채점 = fev 채점 · 정답 · 형식 · 누출")
    c.add_argument("tasks", nargs="+")
    sub.add_parser("reference", help="fev-bench 공개 결과 → results/forecast_reference.csv")
    e = sub.add_parser("elo", help="Elo (Seasonal Naive = 1000) → results/forecast_elo.csv")
    e.add_argument("--rounds", type=int, default=200)
    f = sub.add_parser("fm", help="FM · 기존 방법을 원격 GPU 로 (모든 window, 묶음마다 작업 하나) → fab remote kick")
    f.add_argument("tasks", nargs="+", help="fev task 또는 대회 이름, all = 10개")
    f.add_argument("--methods", nargs="+", help=f"방법 묶음 하나 (없으면 기본 묶음 {len(DEFAULT_GROUPS)}개: "
                   + " / ".join(" ".join(g) for g in DEFAULT_GROUPS) + ")")
    f.add_argument("--accelerator", default=None, help="기본 databricks.local.yaml 의 accelerator (GPU_1xH100)")
    f.add_argument("--timeout", type=int, default=600, help="작업 시간 제한 (분)")
    f.add_argument("--note", default="")
    f.add_argument("--dry-run", action="store_true", help="제출하지 않고 job.json · air.yaml 만 보여 줌")
    sub.add_parser("agent", help="에이전트 실행 (fab fc agent -h, fab mleb agent 와 같은 명령)")
    args = ap.parse_args(argv)

    def names(xs):
        return list(tasks()) if xs == ["all"] else list(dict.fromkeys(resolve(x) for x in xs))

    if args.cmd == "prepare":
        for t in names(args.tasks):
            make_comp(t)
    elif args.cmd == "describe":
        from fab.bench.forecast import describe

        for t in names(args.tasks):
            describe.write(t)
    elif args.cmd == "check":
        ok = True
        for t in names(args.tasks):
            r = check(t)
            print(f"{t}: {json.dumps(r, ensure_ascii=False)}")
            ok &= all(v["max_rel_diff"] < 1e-5 for k, v in r.items() if isinstance(v, dict) and "max_rel_diff" in v) \
                and r["answers_match_fev"] and all(r["invalid_rejected"].values()) and not r["leaks"]
        print("통과" if ok else "실패 있음")
        if not ok:
            sys.exit(1)
    elif args.cmd == "fm":
        ts = names(args.tasks)
        for t in ts:  # 마지막 window 를 채점할 정답 (내 Volume)
            make_comp(t)
        groups = [args.methods] if args.methods else DEFAULT_GROUPS
        if args.dry_run:
            for g in groups:
                r = submit_fm(g, ts, args.accelerator, args.timeout, args.note, dry_run=True)
                print(f"== {r['key']}\n{r['air']}  methods {g} × tasks {list(r['job']['tasks'])}")
            return
        from fab.bench.forecast.remote import upload_answers
        from fab.remote import settings as remote_settings

        upload_answers(remote_settings(), [comp_name(t) for t in ts])
        for g in groups:
            submit_fm(g, ts, args.accelerator, args.timeout, args.note)
        print("끝난 뒤 fab remote kick → 회수 · 마지막 window 채점, fab sync 로 results/forecast.csv")
    elif args.cmd == "reference":
        reference()
    elif args.cmd == "elo":
        elo_tables(args.rounds)
