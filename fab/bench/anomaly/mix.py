"""TFM 을 쓰는 법 (1): 이미 있는 행별 점수 섞기 · 진단으로 고르기 (docs/anomaly.md "TFM 을 쓰는 법 (1)"). 공식 풀 · 기존 결과표 ·
Elo 파일은 안 바꿈: 섞기 · 고르기의 채점은 results/anomaly_mix.csv, 진단값은 anomaly_mix_diag.csv, Elo 는 anomaly_mix_elo.csv.

    uv run --frozen fab ad mix pull                       # Volume 의 outputs/<대회>/ 점수 (test.npy) · 내 train 점수 작업의 train.npy 받기 (읽기만)
    uv run --frozen fab ad mix check                      # 받은 test 점수를 다시 채점 → 기존 결과표와 같은지
    uv run --frozen fab ad mix submit official            # train 행 점수 GPU 작업 계획 (--yes 로 제출. 정리 작업 큐에 기록하지 않음)
    uv run --frozen fab ad mix status                     # 낸 작업의 상태
    uv run --frozen fab ad mix pull --data official e1 s3 # 로컬 CPU 방법의 train 점수에 쓸 train.csv · test.csv (Volume 의 data/)
    uv run --no-project --python $FAB_ANOMALY/.venv/bin/python python -m fab ad mix train official --methods iforest knn ecod   # pyod (옆 가상환경)
    uv run --frozen fab ad mix train s3 --methods lgbm_clf+la100 catboost_clf+la100                                             # GBDT (pyod 와 다른 프로세스)
    uv run --frozen fab ad mix report                     # M1 ~ M3 표 · Elo → results/anomaly_mix*.csv

worktree 에서는 앞에 FAB_OUTPUTS=<worktree>/results (안 붙이면 점수 파일을 main 체크아웃의 results/ 에서 찾음).

실험 (번호는 문서와 같음)
    M1  비지도 TFM (tabpfn_unsup · kumo_unsup) & IForest 섞기: 공식 12 × 3 · E1 합성 46 · E2 94칸
    M2  분류기 (+la<k>) & 비지도 TFM 섞기: S3 40 대회 × test 종류 4 · 공식 36 × la 6
    M3  라벨 없는 진단값 (if10 등) 으로 TFM / IForest 고르기: 임계값은 E1 에서 정해 공식 36 단위에 그대로

분위의 기준 (결정 22): train = 그 방법의 train 행 점수 분포 안의 자리 (주. test 행 하나의 분위가 다른 test 행과 무관), batch = 대회의
test 행 안의 순위 (이름 끝에 @batch). train 행 점수 (<대회>/<방법>/train.npy): TFM · 분류기는 2겹 (train 을 반 나눠 서로를 context ·
학습 행으로), pyod 는 in-sample (decision_scores_) = methods.train_scores (E3) 와 같은 나누기 · 같은 계산. 원격 실행기가
`fab run <대회> <방법> --train-scores` 로 부름 (fab/runner/run.py → train_main). test 점수 (test.npy) · results.csv 는 안 씀.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from fab.bench.anomaly import ANSWERS, CONDITIONS, DATASETS, GRADES, ID_COL, LABEL_COL, SCORE_COL, SPLITS, SUPERVISED
from fab.core.competition import Competition, load_competition
from fab.core.results import OUTPUTS, RESULTS

MIX = RESULTS / "anomaly_mix.csv"  # 섞기 · 고르기의 채점 (로컬 계산: 매번 통째로 다시 씀)
DIAG = RESULTS / "anomaly_mix_diag.csv"  # 단위마다 진단값 · 임계값 · 고른 쪽
MIX_ELO = RESULTS / "anomaly_mix_elo.csv"
RUNS = RESULTS / "anomaly_mix_runs.csv"  # train 행 점수 작업의 실행 이름 (정리 작업 큐 밖: 점수는 Volume 의 runs/<실행>/results/ 에서 직접)
GRIDS = ["official", "e1", "e2", "s3"]
TRAIN_GRIDS = ["official", "e1", "s3"]  # train 행 점수를 내는 격자 (E2 는 context 를 조작한 칸 → 배치 기준만)
TRAIN_NPY, TRAIN_INFO = "train.npy", "train.json"
TFMS = ["tabpfn_unsup", "kumo_unsup"]
MAIN_TFM, PARTNER = "tabpfn_unsup", "iforest"
S3_LA = "+la100"


# --- 단위 (대회) : 기존 결과표에서 -------------------------------------------------------------------------------


def _last(g: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return g.sort_values("timestamp", kind="stable").drop_duplicates(keys, keep="last")


@cache
def graded() -> pd.DataFrame:
    """기존 결과표 셋의 (대회, 방법[, test 종류]) 마다 마지막 행: table · comp · dataset · split · model · group · exp · cond · score · aupr · run.
    섞기에 쓰는 test 점수는 이 표에 있는 실행의 것만 (점수 파일의 run 이 같아야 함, load_test)."""
    off = _last(pd.read_csv(GRADES), ["comp", "model"]).assign(table="anomaly", exp="base", cond="")
    cond = _last(pd.read_csv(CONDITIONS), ["comp", "model"]).assign(table="conditions")
    sup = pd.read_csv(SUPERVISED)
    sup["cond"] = sup["cond"].fillna("")
    sup = _last(sup, ["comp", "model", "cond"]).assign(table="supervised")
    t = pd.concat([off, cond, sup], ignore_index=True)
    t["cond"] = t["cond"].fillna("")
    t["group"] = t.cond.str.extract(r"test=(\w+)")[0].fillna("")
    return t[["table", "comp", "dataset", "split", "model", "group", "exp", "cond", "score", "aupr", "run"]]


def grid_of(comp: str) -> str:
    if comp.startswith(("ad-", "ads-")):
        return "official"
    for g in ("e1", "e2", "s3"):
        if comp.startswith(f"adc-{g}-"):
            return g
    raise ValueError(f"이상탐지 대회가 아님: {comp}")


@cache
def units(grid: str) -> pd.DataFrame:
    """격자의 대회들: comp · dataset · split + 조건 (E1 type, E2 rate · arrange · draw, S3 train). IForest 행이 있는 대회 (모든 격자에 있음)."""
    from fab.bench.anomaly.conditions import parse_cond

    g = graded()
    g = g[(g.model == PARTNER) & (g.comp.map(grid_of) == grid)].drop_duplicates("comp")
    c = g.cond.map(parse_cond)
    for k in {"e1": ["type"], "e2": ["rate", "arrange", "draw"], "s3": ["train"]}.get(grid, []):
        g = g.assign(**{k: c.map(lambda d, k=k: d.get(k))})
    if grid == "e2":
        g = g.assign(rate=g.rate.astype(float))
    return g.drop(columns=["table", "model", "group", "exp", "cond", "score", "aupr", "run"]).sort_values("comp").reset_index(drop=True)


def has_test(comp: str, model: str) -> bool:
    g = graded()
    return bool(((g.comp == comp) & (g.model == model)).any())


def grid_comps(grid: str, need: list[str] | None = None) -> list[Competition]:
    """격자의 대회 (설정만: 데이터 준비 · 정답 쓰기 없음. 데이터는 Volume 의 data/<대회>/). need: 이 방법들의 test 점수가 표에 있는 대회만
    (S3 의 tabpfn_unsup 없는 2칸은 train 점수도 안 냄)."""
    from fab.bench.anomaly import split_comp
    from fab.bench.anomaly.conditions import E1_TYPES, e1_comps
    from fab.bench.anomaly.supervised import s3_comps

    if grid == "official":
        comps = [split_comp(ds, s) for ds in DATASETS for s in SPLITS]
    elif grid == "e1":
        comps = e1_comps(E1_TYPES, list(DATASETS), [1])
    elif grid == "s3":
        comps = s3_comps()
    else:
        raise SystemExit(f"train 행 점수를 내는 격자는 {TRAIN_GRIDS}: {grid}")
    return [c for c in comps if all(has_test(c.name, m) for m in need or [])]


# --- train 행 점수 (원격 GPU · 로컬 CPU 가 같이 씀) --------------------------------------------------------------


def fold_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """train 을 반으로: methods.train_scores (E3 의 정리 점수) 와 같은 나누기."""
    perm = np.random.default_rng([seed, n]).permutation(n)
    return perm[: n // 2], perm[n // 2:]


def train_scores(method: str, X: np.ndarray, seed: int, y: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """train 행의 점수 (클수록 이상) 와 만든 방식. 분류기 (+la<k>, y = PU 라벨): 2겹 교차 (한쪽 반으로 fit → 다른 반의 P(이상)).
    pyod: in-sample (methods.train_scores = decision_scores_). 비지도 FM: 2겹 (서로를 context 로) — methods.train_scores 와 같되
    유한 검사 없이 (밀도가 0 이면 +inf = 맨 위 순위로 쓰고, NaN 은 읽는 쪽이 기준 분포에서 뺌)."""
    from fab.bench.anomaly import methods as M

    if M.is_supervised(method):
        if y is None:
            raise ValueError(f"{method}: 라벨 (y) 이 필요")
        a, b = fold_split(len(X), seed)
        s = np.full(len(X), np.nan)
        for fit, query in ((b, a), (a, b)):
            if y[fit].sum() == 0 or y[fit].all():
                raise ValueError(f"{method}: 한쪽 반에 라벨 준 이상 (또는 정상) 이 없음 ({int(y[fit].sum())} / {len(fit)})")
            s[query] = M.clf_scores(method, X[fit], y[fit], X[query], seed)
        return s, "2fold"
    if not M.is_gpu(method):
        return M.train_scores(method, X, seed), "in-sample"
    raw = M.fomo_scores if M.is_fomo(method) else (M.kumo_scores if M.is_kumo(method) else M.tabpfn_scores)
    a, b = fold_split(len(X), seed)
    s = np.full(len(X), np.nan)
    s[a] = raw(M.base(method), X[b], X[a], seed)
    s[b] = raw(M.base(method), X[a], X[b], seed)
    return s, "2fold"


def nonfinite(s: np.ndarray) -> dict:
    return dict(posinf=int(np.isposinf(s).sum()), neginf=int(np.isneginf(s).sum()), nan=int(np.isnan(s).sum()))


def read_la(labels_dir: Path, comp: str, ids: pd.Series, method: str) -> np.ndarray:
    from fab.bench.anomaly import methods as M

    la = pd.read_csv(Path(labels_dir) / f"{comp}.la.csv").set_index(ID_COL)[f"la{M.la_spec(method)}"]
    return la.loc[ids].to_numpy().astype(int)


def write_train(comp: str, method: str, s: np.ndarray, info: dict):
    out = OUTPUTS / comp / method
    out.mkdir(parents=True, exist_ok=True)
    (out / TRAIN_INFO).unlink(missing_ok=True)  # 저장 도중 멈춰도 옛 출처가 남지 않게
    np.save(out / TRAIN_NPY, s)
    (out / TRAIN_INFO).write_text(json.dumps(info, indent=1))


def train_one(comp: Competition, methods: list[str], labels_dir: str | None, note: str, refit: bool) -> list[str]:
    """대회 하나의 방법들 → <대회>/<방법>/train.npy · train.json. refit (로컬 CPU): 같은 seed 로 다시 fit 한 모형의 test 점수가
    받아 둔 test.npy 와 같은지도 적음 (로컬 계산이 원격의 그 모형을 재현하는지). 실패한 방법 이름을 돌려줌."""
    from fab.bench.anomaly import methods as M

    train = pd.read_csv(comp.data_dir / "train.csv")
    feats = [c for c in train.columns if c != ID_COL]
    X = train[feats].to_numpy(dtype=float)
    seed = int(comp.anomaly["seed"])
    failed = []
    for m in methods:
        t0 = time.time()
        try:
            y = None
            if M.is_supervised(m):
                if not labels_dir:
                    raise RuntimeError("+la 분류기는 --train-labels 가 필요")
                y = read_la(Path(labels_dir), comp.name, train[ID_COL], m)
            s, kind = train_scores(m, X, seed, y)
            if s.shape != (len(X),):
                raise ValueError(f"점수 모양 {s.shape}")
            info = dict(run=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), kind=kind,
                        n=len(s), **nonfinite(s), opts=M.opts(m), note=note)
            if refit and (OUTPUTS / comp.name / m / "test.npy").exists():
                X_te = pd.read_csv(comp.data_dir / "test.csv")[feats].to_numpy(dtype=float)
                again = M.clf_scores(m, X, y, X_te, seed) if M.is_supervised(m) else M.scores(m, X, X_te, seed)
                old = np.load(OUTPUTS / comp.name / m / "test.npy")
                info["refit_max_abs_diff"] = float(np.abs(again - old).max())
                info["refit_auroc_diff"] = auroc_diff(comp.name, again, old)
            info["seconds"] = round(time.time() - t0, 1)
            write_train(comp.name, m, s, info)
            print(f"[mix] {comp.name} {m}: train 점수 {len(s)}행 ({kind}, {info['seconds']}s, 유한하지 않음 "
                  f"{info['posinf']}/{info['neginf']}/{info['nan']}"
                  + (f", 다시 fit 한 test 점수 |차| {info['refit_max_abs_diff']:.2e}" if "refit_max_abs_diff" in info else "") + ")",
                  flush=True)
        except Exception:  # noqa: BLE001 - 방법 하나가 실패해도 나머지는 돌림
            failed.append(m)
            print(f"[mix] {comp.name} {m} 실패:\n{traceback.format_exc()}", flush=True)
    return failed


def auroc_diff(comp: str, a: np.ndarray, b: np.ndarray) -> float | None:
    """같은 대회의 두 test 점수의 AUROC 차 (정답이 로컬에 있을 때만)."""
    from sklearn.metrics import roc_auc_score

    f = ANSWERS / f"{comp}.csv"
    if not f.exists():
        return None
    y = pd.read_csv(f)[LABEL_COL].to_numpy()
    return float(roc_auc_score(y, a) - roc_auc_score(y, b))


def train_main(argv: list[str]) -> int:
    """원격 전용: `fab run <대회> <방법...> --train-scores [--train-labels <Volume 경로>] [--note <글>]` (fab/runner/run.py 가 넘김)."""
    from fab.bench.anomaly import methods as M

    ap = argparse.ArgumentParser(prog="fab run (anomaly, train 행 점수)")
    ap.add_argument("comp")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--train-scores", action="store_true", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--train-labels", default=None)
    args = ap.parse_args(argv)
    if not os.environ.get("FAB_REMOTE"):
        sys.exit(f"원격 전용 (로컬 CPU 방법은 fab ad mix train): {args.comp} {' '.join(args.methods)}")
    if bad := [m for m in args.methods if not M.is_known(m)]:
        sys.exit(f"알 수 없는 방법: {bad}")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    failed = train_one(load_competition(args.comp), args.methods, args.train_labels, args.note, refit=False)
    if failed:
        print(f"[mix] 실패한 방법: {', '.join(failed)}")
        return 1
    return 0


# --- 제출 · 받기 (로컬) -------------------------------------------------------------------------------------------

# 어림 (docs/anomaly.md "train 행 점수"): 2겹 train 점수 = test 한 번의 1.42배 (E3 에서 잰 값), 격자마다 tabpfn_unsup 의 test 한 번 (분)
TEST_MINUTES = {"official": 69.4, "e1": 81.5, "s3": 45.2}
FOLD_FACTOR = 1.42


def read_runs() -> pd.DataFrame:
    cols = ["key", "run_id", "grid", "methods", "comps", "submitted"]
    return pd.read_csv(RUNS, dtype=str) if RUNS.exists() else pd.DataFrame(columns=cols)


def submit(grid: str, methods: list[str], timeout: int, accelerator: str | None, yes: bool, note: str):
    """train 행 점수 GPU 작업 하나 (대회마다 fab run … --train-scores). 정리 작업 큐 (state/fm) 에 기록하지 않음 → 다른 작업의 kick 이
    건드리지 않고 결과표에 행이 안 생김. 실행 이름은 results/anomaly_mix_runs.csv, 점수는 Volume 의 runs/<실행>/results/ 에서 직접."""
    from fab.bench.anomaly import methods as M
    from fab.bench.anomaly.remote import TRAIN_LABELS_REL, train_labels_dir
    from fab.bench.tabarena.remote import _missing
    from fab.runner.databricks import settings as dbx_settings
    from fab.runner.databricks import submit as submit_gpu

    if bad := [m for m in methods if not M.is_known(m) or not M.is_gpu(m)]:
        raise SystemExit(f"GPU 방법이 아님 (로컬 CPU 방법은 fab ad mix train): {bad}")
    comps = grid_comps(grid, need=[MAIN_TFM])
    label = f"adm-{grid}-train"
    unsup = [m for m in methods if not M.is_supervised(m)]
    est = TEST_MINUTES[grid] * FOLD_FACTOR * len(comps) / len(units(grid)) if unsup else 0
    print(f"{label}: 대회 {len(comps)}개 (예: {', '.join(c.name for c in comps[:3])} …)\n  방법 {len(methods)}개: {' '.join(methods)}\n"
          f"  어림: 비지도 2겹 약 {est:.0f}분 + 분류기 교차 점수 몇 분 (H100 1장, 실행기 안. 노드 대기 · 설치는 따로), 시간 제한 {timeout}분")
    if not yes:
        print("계획만 (제출은 --yes)")
        return
    cfg = dbx_settings(argparse.Namespace(accelerator=accelerator, image=None))
    names = [c.name for c in comps]
    if miss := sorted(_missing(cfg, "data", names)):
        raise SystemExit(f"Volume 의 data/ 에 없는 대회 {len(miss)}개 (예: {miss[:3]}) → 올리지 않고 멈춤")
    extra = []
    if any(M.is_supervised(m) for m in methods):
        if miss := sorted(_missing(cfg, TRAIN_LABELS_REL, [f"{n}.la.csv" for n in names])):
            raise SystemExit(f"Volume 의 {TRAIN_LABELS_REL}/ 에 없는 la 라벨 {len(miss)}개 (예: {miss[:3]}) → 올리지 않고 멈춤")
        extra = ["--train-labels", train_labels_dir(cfg)]
    jobs = [(n, f"{cfg['volume']}/data/{n}", [*methods, "--train-scores", "--note", note or label, *extra]) for n in names]
    run = submit_gpu(cfg, label, None, None, methods, cfg["accelerator"], timeout, "", jobs=jobs)
    row = dict(key=run["key"], run_id=str(run["run_id"]), grid=grid, methods=" ".join(methods), comps=str(len(comps)),
               submitted=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"))
    pd.concat([read_runs(), pd.DataFrame([row])], ignore_index=True).to_csv(RUNS, index=False)
    print(f"제출: {run['key']} (run_id {run['run_id']}) → {RUNS.name} 에 기록 (정리 작업 큐에는 기록 안 함)\n  {run['url']}")


def status():
    from fab.remote import settings
    from fab.runner.databricks import status as run_status

    cfg = settings()
    for r in read_runs().itertuples():
        try:
            st = run_status(cfg["profile"], r.run_id)
        except Exception as e:  # noqa: BLE001
            st = f"UNKNOWN ({e!r})"[:120]
        print(f"{r.key} (run_id {r.run_id}, {r.grid}: {r.methods}): {st}")


def _pull(cfg, rel: str, dst: Path) -> tuple[str, int, str]:
    from fab.remote import dbx

    dst.parent.mkdir(parents=True, exist_ok=True)
    r = dbx(cfg, "fs", "cp", "-r", "--overwrite", f"dbfs:{cfg['volume']}/{rel}", str(dst), check=False, timeout=3600)
    return rel, r.returncode, r.stderr.strip()[-200:]


def pull(data: list[str], runs: bool, outputs: bool, workers: int):
    """Volume → 로컬 (읽기만, 결과표는 안 받음). outputs: 이상탐지 대회의 outputs/<대회>/ (test.npy · test.json) → OUTPUTS.
    runs: 내 train 점수 작업의 runs/<실행>/results/<대회>/<방법>/train.* → OUTPUTS. data: 격자의 data/<대회>/{train,test}.csv →
    competitions/<대회>/data/ (로컬 CPU 방법의 train 점수용)."""
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from fab.remote import list_names, settings

    cfg = settings()
    bad = []

    def many(pairs):
        with ThreadPoolExecutor(workers) as ex:
            for rel, rc, err in ex.map(lambda p: _pull(cfg, *p), pairs):
                if rc != 0:
                    bad.append((rel, err))

    if outputs:
        comps = sorted(n.strip("/") for n in list_names(cfg, "outputs") if n.startswith(("ad-", "ads-", "adc-")))
        print(f"outputs/: 이상탐지 대회 {len(comps)}개 → {OUTPUTS}")
        many([(f"outputs/{c}", OUTPUTS / c) for c in comps])
    for grid in data:
        comps = grid_comps(grid)
        print(f"data/: {grid} 대회 {len(comps)}개 → competitions/<대회>/data/")
        many([(f"data/{c.name}/{f}", c.data_dir / f) for c in comps for f in ("train.csv", "test.csv")
              if not (c.data_dir / f).exists()])
    if runs:
        for r in read_runs().itertuples():
            with tempfile.TemporaryDirectory() as d:
                rel, rc, err = _pull(cfg, f"runs/{r.key}/results", Path(d) / "results")
                if rc != 0:
                    bad.append((rel, err))
                    continue
                n = 0
                for f in sorted((Path(d) / "results").glob(f"*/*/{TRAIN_NPY}")):
                    out = OUTPUTS / f.parent.parent.name / f.parent.name
                    out.mkdir(parents=True, exist_ok=True)
                    for name in (TRAIN_INFO, TRAIN_NPY):
                        if (f.parent / name).exists():
                            shutil.copyfile(f.parent / name, out / name)
                    n += 1
                print(f"runs/{r.key}: train 점수 {n}개 ({r.grid}: {r.methods})")
    if bad:
        print(f"받기 실패 {len(bad)}개: {bad[:5]}")


def train_local(grid: str, methods: list[str], labels_dir: Path):
    """로컬 CPU 방법 (pyod · GBDT) 의 train 행 점수. 데이터는 fab ad mix pull --data 로 받은 것. pyod 와 GBDT 는 다른 프로세스로
    (macOS 에서 LightGBM 과 torch 를 한 프로세스에 두면 segfault, docs/known-issues.md)."""
    from fab.bench.anomaly import methods as M

    if bad := [m for m in methods if not M.is_known(m) or (M.is_gpu(m) and not os.environ.get("FAB_MIX_GPU_LOCAL"))]:
        raise SystemExit(f"로컬 CPU 방법이 아님 (GPU 방법은 fab ad mix submit): {bad}")
    comps = grid_comps(grid, need=[MAIN_TFM] if grid == "s3" else None)
    if miss := [c.name for c in comps if not (c.data_dir / "train.csv").exists()]:
        raise SystemExit(f"데이터 없는 대회 {len(miss)}개 (예: {miss[:3]}) → fab ad mix pull --data {grid}")
    failed = {}
    for c in comps:
        if f := train_one(c, methods, str(labels_dir), "local", refit=True):
            failed[c.name] = f
    print(f"{grid}: 대회 {len(comps)}개 × 방법 {len(methods)}개" + (f", 실패 {failed}" if failed else ", 실패 0"))


# --- 점수 읽기 · 채점 --------------------------------------------------------------------------------------------


@cache
def _runs_of() -> dict:
    g = graded()
    return {(c, m): str(r) for c, m, r in zip(g.comp, g.model, g.run)}


@cache
def load_test(comp: str, model: str) -> np.ndarray | None:
    """기존 결과표에 있는 실행의 test 점수 (test.csv 행 순서). 점수 파일의 run 이 표의 run 과 다르면 (그 뒤 다시 돌린 점수) 안 씀."""
    d = OUTPUTS / comp / model
    if (comp, model) not in _runs_of() or not (d / "test.npy").exists() or not (d / "test.json").exists():
        return None
    if str(json.loads((d / "test.json").read_text()).get("run")) != _runs_of()[(comp, model)]:
        return None
    return np.load(d / "test.npy")


@cache
def load_train(comp: str, model: str) -> np.ndarray | None:
    """train 행 점수 (train.csv 행 순서, 이 작업이 낸 것). NaN 은 뺌 (기준 분포에 안 넣음), +inf 는 그대로 (맨 위)."""
    f = OUTPUTS / comp / model / TRAIN_NPY
    if not f.exists():
        return None
    s = np.load(f)
    return s[~np.isnan(s)]


@cache
def answers(comp: str) -> pd.DataFrame:
    """정답 (id · label [· group]). 행 순서 = test.csv 행 순서 (prepare 가 같은 표에서 둘을 씀, fab ad mix check 로 확인)."""
    return pd.read_csv(ANSWERS / f"{comp}.csv")


def groups_of(comp: str) -> list[str]:
    """S3 대회의 test 이상 종류 (정답의 group), 그 밖은 [''] (test 전체로 한 번 채점)."""
    ans = answers(comp)
    return sorted(set(ans["group"]) - {"normal"}) if "group" in ans else [""]


def grade(comp: str, s: np.ndarray, group: str = "") -> dict:
    """기존 채점 함수 (score_frame) 로 → {score (AUROC), aupr}. group: S3 의 test 종류 (정상 + 그 종류 이상만, 기존 grade 와 같음)."""
    from fab.bench.anomaly import score_frame

    ans = answers(comp)
    sub = pd.DataFrame({ID_COL: ans[ID_COL], SCORE_COL: s})
    if group:
        ans = ans[(ans[LABEL_COL] == 0) | (ans["group"] == group)]
        sub = sub[sub[ID_COL].isin(ans[ID_COL])]
    r = score_frame(sub, ans[[ID_COL, LABEL_COL]])
    if not r.get("valid_submission"):
        raise ValueError(f"{comp}: 채점 실패 {r.get('error')}")
    return dict(score=r["score"], aupr=r["aupr"])


def check(tol: float = 1e-9) -> bool:
    """받은 test 점수를 다시 채점 → 기존 결과표의 값 · run 과 같은지. 섞기의 입력이 표의 그 실행이라는 확인."""
    g = graded()
    rows, missing, stale = [], [], []
    for (comp, model), part in g.groupby(["comp", "model"], sort=False):
        d = OUTPUTS / comp / model
        if not (d / "test.npy").exists():
            missing.append(f"{comp}/{model}")
            continue
        s = load_test(comp, model)
        if s is None:
            stale.append(f"{comp}/{model}")
            continue
        for r in part.itertuples():
            new = grade(comp, s, r.group)
            rows.append(dict(table=r.table, comp=comp, model=model, group=r.group, d_auroc=abs(new["score"] - r.score),
                             d_aupr=abs(new["aupr"] - r.aupr)))
    t = pd.DataFrame(rows)
    extra = sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in OUTPUTS.glob("ad*/*/test.npy")
                   if (p.parent.parent.name, p.parent.name) not in _runs_of())
    print(f"결과표 행 {len(g)} ({g.groupby('table').size().to_dict()}), 다시 채점 {len(t)}행")
    if len(t):
        print(f"  |차| 최대: AUROC {t.d_auroc.max():.2e} · AUPRC {t.d_aupr.max():.2e}, {tol:g} 넘는 행 {int(((t.d_auroc > tol) | (t.d_aupr > tol)).sum())}")
    print(f"  점수 파일 없음 {len(missing)}{' ' + str(missing[:5]) if missing else ''}, run 이 표와 다름 {len(stale)}"
          f"{' ' + str(stale[:5]) if stale else ''}, 표에 없는 점수 파일 {len(extra)}{' ' + str(extra[:5]) if extra else ''}")
    return bool(len(t)) and not missing and not stale and float(max(t.d_auroc.max(), t.d_aupr.max())) <= tol


# --- 분위 · 섞는 규칙 · 진단값 ------------------------------------------------------------------------------------

RULES = ["mean", "max", "min", "w25", "w75", "tail"]
MAIN_RULE = "mean"  # 결정 21: 순위 분위의 평균, 가중치 1 : 1
BASES = ["train", "batch"]  # 결정 22: train 기준이 주
BATCH = "@batch"
KS = [5, 10, 20]
MAIN_DIAG = "if10"  # 결정 23: IForest 상위 10% 행의 TFM 분위 (낮으면 IForest)
DIAGS = ["rho", *(f"{d}{k}" for d in ("if", "tfm", "knn") for k in KS)]
WORST = ["cover", "cardio", "PageBlocks"]  # TFM 이 무너지는 셋 (공식 결과에서 이미 정해진 것)
WIN = ["Wilt", "annthyroid", "landsat", "magic.gamma"]  # TFM 이 1등인 넷
CLFS = ["tabpfn_clf", "kumo_clf", "lgbm_clf", "catboost_clf"]
MAIN_CLF = "tabpfn_clf"
S3_TYPES = ["local", "global", "dependency", "cluster"]


def q_batch(s: np.ndarray) -> np.ndarray:
    """그 행들 안의 순위 분위 (순위 − 0.5) / n, 같은 점수는 평균 순위. 배치 기준의 분위이자 진단값이 쓰는 분위."""
    from scipy.stats import rankdata

    return (rankdata(s, method="average") - 0.5) / len(s)


def q_ref(s: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """기준 점수 ref (n 개, NaN 없음) 안에서 s 의 자리: (작은 개수 + 같은 개수 / 2 + 0.5) / (n + 1). train 기준의 분위
    (행 하나의 분위가 다른 test 행과 무관). 0 과 1 사이 (끝 값 제외)."""
    r = np.sort(ref)
    lt, le = np.searchsorted(r, s, side="left"), np.searchsorted(r, s, side="right")
    return (lt + 0.5 * (le - lt) + 0.5) / (len(r) + 1)


def combine(qa: np.ndarray, qb: np.ndarray, rule: str) -> np.ndarray:
    if rule == "mean":
        return (qa + qb) / 2
    if rule == "max":
        return np.maximum(qa, qb)
    if rule == "min":
        return np.minimum(qa, qb)
    if rule in ("w25", "w75"):
        w = int(rule[1:]) / 100
        return w * qa + (1 - w) * qb
    if rule == "tail":  # 위쪽 꼬리를 키운 합 (q < 1 이라 유한)
        return -np.log1p(-qa) - np.log1p(-qb)
    raise ValueError(f"규칙은 {RULES}: {rule}")


def top_mean(q_pick: np.ndarray, q_other: np.ndarray, k: float) -> float:
    """q_pick 상위 k% 행의 q_other 평균을 [−1, 1] 로: 1 = q_other 의 상위 k% 와 같은 행, 0 = 무관, 음수 = 보통 행보다 낮음."""
    m = max(1, int(round(k / 100 * len(q_pick))))
    top = np.argsort(-q_pick, kind="stable")[:m]
    return float((q_other[top].mean() - 0.5) / (0.5 - m / len(q_pick) / 2))


def diag_values(s_tfm: np.ndarray, s_if: np.ndarray, s_knn: np.ndarray | None = None) -> dict:
    """라벨 없는 진단값 (같은 행들의 점수 → 그 행들 안의 순위 분위로). NaN 이 있는 행은 뺌 (+inf 는 맨 위 순위)."""
    ok = ~(np.isnan(s_tfm) | np.isnan(s_if) | (np.isnan(s_knn) if s_knn is not None else False))
    qt, qi = q_batch(s_tfm[ok]), q_batch(s_if[ok])
    out = dict(rho=float(np.corrcoef(qt, qi)[0, 1]))
    qk = q_batch(s_knn[ok]) if s_knn is not None else None
    for k in KS:
        out[f"if{k}"] = top_mean(qi, qt, k)
        out[f"tfm{k}"] = top_mean(qt, qi, k)
        out[f"knn{k}"] = top_mean(qi, qk, k) if qk is not None else np.nan
    return out


def fit_tau(d: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> tuple[float, float]:
    """"진단값 ≥ τ 면 hi, 아니면 lo" 로 골랐을 때 평균이 가장 높은 τ 와 그 평균. 후보 = 정렬한 진단값 사이의 중간점과 양 끝
    (−inf = 늘 hi, +inf = 늘 lo), 같은 평균이 여럿이면 가운데 후보 (결정 24)."""
    u = np.unique(d)
    cands = [-np.inf, *((u[:-1] + u[1:]) / 2), np.inf]
    vals = np.array([np.where(d >= t, hi, lo).mean() for t in cands])
    best = np.flatnonzero(vals >= vals.max() - 1e-12)
    return float(cands[best[len(best) // 2]]), float(vals.max())


# --- 점수 → 분위 (기준마다) ---------------------------------------------------------------------------------------


@cache
def load_train_raw(comp: str, model: str) -> np.ndarray | None:
    f = OUTPUTS / comp / model / TRAIN_NPY
    return np.load(f) if f.exists() else None


@cache
def quantiles(comp: str, model: str, basis: str) -> np.ndarray | None:
    """test 행의 분위. batch: test 행 안의 순위. train: 그 방법의 train 행 점수 안의 자리 (train 점수가 없으면 None)."""
    s = load_test(comp, model)
    if s is None:
        return None
    if basis == "batch":
        return q_batch(s)
    ref = load_train(comp, model)
    return q_ref(s, ref) if ref is not None and len(ref) else None


def mix_name(a: str, b: str, rule: str, basis: str) -> str:
    return f"{a}&{b}:{rule}" + (BATCH if basis == "batch" else "")


def pick_name(hi: str, lo: str, diag: str, basis: str) -> str:
    return f"{hi}/{lo}:{diag}" + (BATCH if basis == "batch" else "")


def mix_rows(exp: str, grid: str, pairs: list[tuple[str, str]], rules: list[str], bases: list[str],
             cond_of=lambda u, g: "") -> list[dict]:
    """격자의 단위마다 짝 × 규칙 × 기준 → 채점 행. 짝의 어느 쪽이든 그 기준의 분위가 없으면 (점수 파일 없음) 그 단위는 건너뜀.
    S3 는 test 종류마다 한 행 (분위는 test 행 전부 또는 train 기준에서, AUROC 는 정상 + 그 종류)."""
    rows = []
    for u in units(grid).itertuples():
        for a, b in pairs:
            for basis in bases:
                qa, qb = quantiles(u.comp, a, basis), quantiles(u.comp, b, basis)
                if qa is None or qb is None:
                    continue
                for rule in rules:
                    s = combine(qa, qb, rule)
                    for g in groups_of(u.comp):
                        r = grade(u.comp, s, g)
                        rows.append(dict(exp=exp, grid=grid, comp=u.comp, dataset=u.dataset, split=u.split, cond=cond_of(u, g),
                                         model=mix_name(a, b, rule, basis), basis=basis, rule=rule, a=a, b=b, score=r["score"],
                                         metric_error=1 - r["score"], aupr=r["aupr"]))
    return rows


def cond_text(grid: str):
    if grid == "e1":
        return lambda u, g: f"type={u.type}"
    if grid == "e2":
        return lambda u, g: f"rate={u.rate:g} arrange={u.arrange} draw={u.draw}"
    if grid == "s3":
        return lambda u, g: f"train={u.train} test={g}"
    return lambda u, g: ""


def singles(grid: str) -> pd.DataFrame:
    """기존 결과표의 단독 방법 AUROC (다시 계산하지 않음): comp · model · group · score · aupr."""
    g = graded()
    return g[g.comp.map(grid_of) == grid][["comp", "dataset", "split", "model", "group", "score", "aupr"]]


def diag_rows(grid: str, tfm: str, basis: str) -> list[dict]:
    """단위마다 진단값 (basis = train: train 행 점수로, batch: test 행 점수로) + 두 단독의 AUROC."""
    rows = []
    sc = singles(grid).set_index(["comp", "model"]).score
    for u in units(grid).itertuples():
        get = load_train_raw if basis == "train" else load_test
        st, si, sk = get(u.comp, tfm), get(u.comp, PARTNER), get(u.comp, "knn")
        if st is None or si is None or load_test(u.comp, tfm) is None:
            continue
        rows.append(dict(grid=grid, comp=u.comp, dataset=u.dataset, split=u.split, cond=cond_text(grid)(u, ""), tfm=tfm,
                         basis=basis, **diag_values(st, si, sk), auroc_tfm=sc[(u.comp, tfm)], auroc_if=sc[(u.comp, PARTNER)]))
    return rows

# --- 보고 ------------------------------------------------------------------------------------------------------

M1_PAIRS = [(MAIN_TFM, PARTNER), ("kumo_unsup", PARTNER),  # 주 · 복제
            (MAIN_TFM, "kumo_unsup"), (PARTNER, "knn"), (PARTNER, "ecod")]  # 대조: TFM 끼리, 기존 방법끼리


def wide(rows: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    return rows.pivot_table(index="comp", columns="model", values=value, aggfunc="first")


def by_dataset(s: pd.Series, grid: str = "official") -> pd.Series:
    """단위 (대회) 별 값 → 데이터셋 평균 (split 평균)."""
    ds = units(grid).set_index("comp").dataset
    return s.groupby(ds.reindex(s.index).to_numpy()).mean()


def summary(S: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """공식 36 단위의 방법들 (cols = {표시 이름: S 의 열}) → 12개 평균 · 꼴찌 셋 (평균 · 최저) · 이기는 넷 · IForest 대비 가장 큰 손해."""
    base = by_dataset(S[PARTNER])
    out = []
    for label, c in cols.items():
        if c not in S or S[c].isna().all():
            continue
        m = by_dataset(S[c].dropna())
        d = (m - base).dropna()
        out.append(dict(method=label, n=len(m), mean=m.mean(), worst3=m.reindex(WORST).mean(), worst_min=m.reindex(WORST).min(),
                        win4=m.reindex(WIN).mean(), max_loss=d.min(), loss_ds=d.idxmin(), ge_if=int((d >= 0).sum())))
    return pd.DataFrame(out).set_index("method")


def position(S: pd.DataFrame, mix: str, tfm: str) -> pd.DataFrame:
    """꼴찌 셋의 되찾은 몫 (섞기 − TFM) / (IForest − TFM), 이기는 넷의 지킨 몫 (섞기 − IForest) / (TFM − IForest). 데이터셋 평균으로."""
    m, t, i = by_dataset(S[mix].dropna()), by_dataset(S[tfm]), by_dataset(S[PARTNER])
    rows = [dict(dataset=d, group="꼴찌 셋", tfm=t[d], iforest=i[d], mix=m.get(d), pos=(m.get(d, np.nan) - t[d]) / (i[d] - t[d]))
            for d in WORST]
    rows += [dict(dataset=d, group="이기는 넷", tfm=t[d], iforest=i[d], mix=m.get(d), pos=(m.get(d, np.nan) - i[d]) / (t[d] - i[d]))
             for d in WIN]
    return pd.DataFrame(rows).set_index("dataset")


def halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng([seed, n, 7]).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def grade_rows(comp: str, idx: np.ndarray, s: np.ndarray) -> float | None:
    """test 행의 일부 (idx) 만으로 AUROC (교차 절반). 한쪽 라벨만 있으면 None."""
    from fab.bench.anomaly import score_frame

    ans = answers(comp).iloc[idx]
    if ans[LABEL_COL].nunique() < 2:
        return None
    r = score_frame(pd.DataFrame({ID_COL: ans[ID_COL].to_numpy(), SCORE_COL: s}), ans[[ID_COL, LABEL_COL]])
    return r["score"]


def cross_half(comp: str, a: str, b: str, rule: str, seed: int) -> tuple[float, float] | None:
    """교차 절반 민감도: test 행을 반으로 → (다른 절반의 점수 분포를 기준으로 한 분위로 섞은 AUROC, 같은 절반 안의 순위로 섞은 AUROC),
    두 절반의 평균. 기준 행과 채점 행이 겹치지 않을 때 배치 기준과 얼마나 다른지."""
    sa, sb = load_test(comp, a), load_test(comp, b)
    if sa is None or sb is None:
        return None
    h = halves(len(sa), seed)
    cross, within = [], []
    for me, other in ((h[0], h[1]), (h[1], h[0])):
        c = grade_rows(comp, me, combine(q_ref(sa[me], sa[other]), q_ref(sb[me], sb[other]), rule))
        w = grade_rows(comp, me, combine(q_batch(sa[me]), q_batch(sb[me]), rule))
        if c is None or w is None:
            return None
        cross.append(c)
        within.append(w)
    return float(np.mean(cross)), float(np.mean(within))


def show(title: str, df: pd.DataFrame | pd.Series, digits: int = 3):
    print(f"\n== {title}\n{df.round(digits).to_string()}")


def report_m1(mix: pd.DataFrame):
    off = mix[(mix.exp == "m1") & (mix.grid == "official")]
    S = wide(off).join(wide(singles("official")), how="outer")
    for tfm in TFMS:
        for basis in BASES:
            name = mix_name(tfm, PARTNER, MAIN_RULE, basis)
            if name in S:
                show(f"M1 {name}: 몫 (꼴찌 셋 = 되찾은 몫, 이기는 넷 = 지킨 몫. 0.5 = 동전 던지기)", position(S, name, tfm))
        cols = {"늘 IForest": PARTNER, f"늘 {tfm}": tfm}
        for basis in BASES:
            cols |= {mix_name(tfm, PARTNER, r, basis): mix_name(tfm, PARTNER, r, basis) for r in RULES}
        S["_oracle"] = S[[tfm, PARTNER]].max(axis=1)
        show(f"M1 공식 12개 × 3 split: {tfm} & IForest (데이터셋 평균. 상한 = 단위마다 test 라벨로 높은 쪽)",
             summary(S, cols | {"상한 (라벨)": "_oracle"}))
        ds = pd.DataFrame({c: by_dataset(S[c].dropna()) for c in
                           [PARTNER, tfm, *(mix_name(tfm, PARTNER, r, b) for b in BASES for r in ("mean", "max"))] if c in S})
        show(f"M1 데이터셋별 AUROC ({tfm})", ds.reindex([*WIN, "fault", "mammography", "yeast", "Waveform", "shuttle", *WORST]))
    ctrl = {"늘 IForest": PARTNER}
    for a, b in M1_PAIRS[2:]:
        ctrl |= {mix_name(a, b, r, bs): mix_name(a, b, r, bs) for bs in BASES for r in ("mean", "max")}
    show("M1 대조: TFM 끼리 · 기존 방법끼리 섞기", summary(S, ctrl))
    # train 기준 − 배치 기준 (둘 다 있는 짝)
    rows = []
    for a, b in M1_PAIRS:
        for r in RULES:
            tr, bt = mix_name(a, b, r, "train"), mix_name(a, b, r, "batch")
            if tr in S and bt in S:
                d = (S[tr] - S[bt]).dropna()
                rows.append(dict(pair=f"{a}&{b}", rule=r, n=len(d), mean_diff=d.mean(), mean_abs=d.abs().mean(), max_abs=d.abs().max()))
    if rows:
        show("M1 공식: train 기준 − 배치 기준 (단위별 AUROC 차)", pd.DataFrame(rows).set_index(["pair", "rule"]), 4)

    # E1: 종류마다
    e1 = mix[(mix.exp == "m1") & (mix.grid == "e1")]
    if len(e1):
        E = wide(e1).join(wide(singles("e1")), how="outer")
        typ = units("e1").set_index("comp").type.reindex(E.index)
        for tfm in TFMS:
            cols = [PARTNER, tfm, *(mix_name(tfm, PARTNER, r, b) for b in BASES for r in RULES)]
            tb = E[[c for c in cols if c in E]].groupby(typ.to_numpy()).mean().T
            show(f"M1 E1 합성 (종류별 데이터셋 평균): {tfm} & IForest", tb[["local", "global", "dependency", "cluster"]])
            for basis in BASES:
                name = mix_name(tfm, PARTNER, MAIN_RULE, basis)
                if name in tb.index:
                    c, d = tb["cluster"], tb["dependency"]
                    print(f"  {name}: cluster 되찾은 몫 {(c[name] - c[tfm]) / (c[PARTNER] - c[tfm]):.3f}, "
                          f"dependency 지킨 몫 {(d[name] - d[PARTNER]) / (d[tfm] - d[PARTNER]):.3f}")
            # 합성에서 고른 변형 → 실제 36 단위
            for basis in BASES:
                names = {r: mix_name(tfm, PARTNER, r, basis) for r in RULES}
                have = {r: n for r, n in names.items() if n in E and n in S}
                if have:
                    best = max(have, key=lambda r: E[have[r]].mean())
                    print(f"  E1 평균이 가장 높은 변형 ({basis}): {best} (E1 {E[have[best]].mean():.4f}) → 실제 12개 평균 "
                          f"{by_dataset(S[have[best]].dropna()).mean():.4f} (주 규칙 {by_dataset(S[have[MAIN_RULE]].dropna()).mean():.4f})")

    # E2: r = 0 대비 떨어진 AUROC (배치 기준)
    e2 = mix[(mix.exp == "m1") & (mix.grid == "e2")]
    if len(e2):
        U = units("e2").set_index("comp")
        W = wide(e2).join(wide(singles("e2")), how="outer")
        r0 = W[U.rate.reindex(W.index).to_numpy() == 0]
        r0.index = U.dataset.reindex(r0.index).to_numpy()
        cells = W[U.rate.reindex(W.index).to_numpy() > 0]
        drop = pd.DataFrame(r0.reindex(U.dataset.reindex(cells.index).to_numpy()).to_numpy() - cells.to_numpy(),
                            index=cells.index, columns=cells.columns)
        arr = U.arrange.reindex(cells.index).to_numpy()
        cols = [PARTNER, *TFMS, *(mix_name(t_, PARTNER, r, "batch") for t_ in TFMS for r in ("mean", "max"))]
        show("M1 E2 94칸: r = 0 대비 떨어진 AUROC (칸 평균, 모임 / 흩어짐)",
             drop[[c for c in cols if c in drop]].groupby(arr).mean().T.rename(columns={"clump": "모임", "spread": "흩어짐"}))
        show("M1 E2: 칸 평균 AUROC (r > 0) 와 r = 0", pd.DataFrame({"r>0": cells[[c for c in cols if c in cells]].mean(),
                                                                    "r=0": r0[[c for c in cols if c in r0]].mean()}))


def s3_table(rows: pd.DataFrame) -> pd.DataFrame:
    """S3 의 (대회, test 종류) 행들 (score · model · train · test) → 방법마다 대각선 · 대각선 밖 요약 (칸 = 데이터셋 평균)."""
    out = []
    for m, g in rows.groupby("model", sort=False):
        cell = g.pivot_table(index="train", columns="test", values="score", aggfunc="mean").reindex(index=S3_TYPES, columns=S3_TYPES)
        diag = np.diag(cell.to_numpy())
        off = cell.to_numpy()[~np.eye(4, dtype=bool)]
        no_global = cell.drop(columns="global").to_numpy()[~np.eye(4, dtype=bool)[:, [0, 2, 3]]]
        out.append(dict(model=m, comps=g.comp.nunique(), diag=diag.mean(), diag_min=diag.min(), off=off.mean(), off_min=off.min(),
                        off_no_global=no_global.mean(), **{"cl>cl": cell.loc["cluster", "cluster"], "cl>dep": cell.loc["cluster", "dependency"],
                                                          "cl>loc": cell.loc["cluster", "local"], "gl>dep": cell.loc["global", "dependency"],
                                                          "loc>dep": cell.loc["local", "dependency"], "dep>loc": cell.loc["dependency", "local"],
                                                          "dep>dep": cell.loc["dependency", "dependency"]}))
    return pd.DataFrame(out).set_index("model")


def report_m2(mix: pd.DataFrame):
    from fab.bench.anomaly.conditions import parse_cond
    from fab.bench.anomaly.methods import LA_LEVELS

    m2 = mix[(mix.exp == "m2") & (mix.grid == "s3")]
    if len(m2):
        c = m2.cond.map(parse_cond)
        m2 = m2.assign(train=c.map(lambda d: d["train"]), test=c.map(lambda d: d["test"]))
        sg = singles("s3")
        tr = units("s3").set_index("comp").train
        sg = sg.assign(train=tr.reindex(sg.comp).to_numpy(), test=sg.group)
        for unsup in TFMS:
            comps = set(m2[m2.b == unsup].comp)
            if not comps:
                continue
            keep = [f"{c_}{S3_LA}" for c_ in CLFS] + [unsup, PARTNER]
            base = sg[sg.comp.isin(comps) & sg.model.isin(keep)]
            pair = m2[(m2.b == unsup) | ((m2.a == f"{MAIN_CLF}{S3_LA}") & (m2.b == PARTNER) & m2.comp.isin(comps))]
            show(f"M2 S3 (대회 {len(comps)}개 × test 종류 4): 분류기 & {unsup} (칸 = 데이터셋 평균. diag = 본 종류, off = 처음 보는 종류)",
                 s3_table(pd.concat([base, pair[pair.rule.isin(["mean", "max"])]], ignore_index=True)))
            main = pair[(pair.a == f"{MAIN_CLF}{S3_LA}") & (pair.b == unsup)]
            if main.rule.nunique() > 2:
                show(f"M2 S3 주 짝 {MAIN_CLF}{S3_LA}&{unsup}: 규칙 민감도", s3_table(main))
            for basis in BASES:
                name = mix_name(f"{MAIN_CLF}{S3_LA}", unsup, MAIN_RULE, basis)
                g = pd.concat([base[base.model.isin([f"{MAIN_CLF}{S3_LA}", unsup])], pair[pair.model == name]])
                if (g.model == name).any():
                    cell = g.pivot_table(index=["train", "test"], columns="model", values="score", aggfunc="mean")
                    show(f"M2 S3 칸별 (데이터셋 평균): {name}", cell.reindex([(a, b) for a in S3_TYPES for b in S3_TYPES]))
        d = m2.pivot_table(index=["comp", "test", "a", "b", "rule"], columns="basis", values="score")
        if {"train", "batch"} <= set(d.columns):
            d = (d["train"] - d["batch"]).dropna()
            show("M2 S3: train 기준 − 배치 기준 (대회 × test 종류의 AUROC 차)",
                 d.groupby(["a", "b", "rule"]).agg(["size", "mean", lambda s: s.abs().mean(), lambda s: s.abs().max()])
                 .rename(columns={"size": "n", "<lambda_0>": "mean_abs", "<lambda_1>": "max_abs"}), 4)

    o = mix[(mix.exp == "m2") & (mix.grid == "official")]
    if len(o):
        sg = singles("official").set_index(["comp", "model"]).score
        o = o.assign(la=o.cond.str.extract(r"la=(\d+)")[0].astype(int), clf=o.a.str.replace(r"\+la\d+$", "", regex=True))
        o = o.assign(alone=[sg[(c_, a)] for c_, a in zip(o.comp, o.a)])
        ds = o.groupby(["clf", "la", "rule", "dataset"]).agg(mix=("score", "mean"), alone=("alone", "mean")).reset_index()
        t = ds.groupby(["clf", "la", "rule"]).agg(alone=("alone", "mean"), mix=("mix", "mean"),
                                                  better=("mix", lambda s: 0)).reset_index()
        t["better"] = ds.assign(b=ds.mix > ds.alone).groupby(["clf", "la", "rule"]).b.sum().to_numpy()
        t["diff"] = t.mix - t.alone
        unsup = by_dataset(singles("official").query("model == @MAIN_TFM").set_index("comp").score).mean()
        show(f"M2 공식 12개 × 3 split × la (배치 기준, 12개 평균. 비지도 {MAIN_TFM} 단독 {unsup:.3f}. better = 섞기가 더 높은 데이터셋 수)",
             t.pivot(index=["clf", "la"], columns="rule", values=["alone", "mix", "diff", "better"])
             .reindex([(c_, k) for c_ in CLFS for k in LA_LEVELS])[[("alone", "mean"), ("mix", "mean"), ("diff", "mean"), ("better", "mean"),
                                                                   ("mix", "max"), ("diff", "max"), ("better", "max")]])
        rows = []
        for u in units("official").itertuples():
            for k in LA_LEVELS:
                r = cross_half(u.comp, f"{MAIN_CLF}+la{k}", MAIN_TFM, MAIN_RULE, int(u.split[1:]))
                if r:
                    rows.append(dict(la=k, cross=r[0], within=r[1]))
        if rows:
            x = pd.DataFrame(rows)
            x["d"] = x.cross - x.within
            show(f"M2 공식 교차 절반 ({MAIN_CLF}&{MAIN_TFM}:{MAIN_RULE}): 다른 절반 기준 − 같은 절반 안 (단위별 AUROC 차)",
                 x.groupby("la").agg(n=("d", "size"), cross=("cross", "mean"), within=("within", "mean"), mean_diff=("d", "mean"),
                                     mean_abs=("d", lambda s: s.abs().mean()), max_abs=("d", lambda s: s.abs().max())), 4)


def pick_scores(D: pd.DataFrame, S: pd.DataFrame, diag: str, sign: int, tau: float, hi: str, lo: str) -> pd.Series:
    """단위마다 고른 방법의 AUROC: sign × 진단값 ≥ τ 면 hi, 아니면 lo."""
    use_hi = sign * D[diag] >= tau
    return pd.Series(np.where(use_hi, S[hi].reindex(D.index), S[lo].reindex(D.index)), index=D.index)


def report_m3(mix: pd.DataFrame, diag: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """진단값 → 임계값 (E1) → 공식 36 단위의 고르기. 돌려주는 것: (임계값 · 고른 쪽을 채운 진단 표, 고르기의 채점 행)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    pick_rows = []
    diag = diag.assign(tau_e1=np.nan, pick_e1="", tau_lodo=np.nan, pick_lodo="")
    S_all = {g: wide(mix[(mix.exp == "m1") & (mix.grid == g)]).join(wide(singles(g)), how="outer") for g in ("official", "e1")}
    aupr_all = {g: wide(mix[(mix.exp == "m1") & (mix.grid == g)], "aupr").join(wide(singles(g), "aupr"), how="outer")
                for g in ("official",)}
    for tfm in TFMS:
        for basis in BASES:
            D = diag[(diag.tfm == tfm) & (diag.basis == basis)]
            De, Do = D[D.grid == "e1"].set_index("comp"), D[D.grid == "official"].set_index("comp")
            if De.empty or Do.empty:
                continue
            Se, So = S_all["e1"], S_all["official"]
            tag = f"{tfm} · {basis} 기준"
            show(f"M3 진단값 ({tag}): 공식 12개 (split 평균), TFM − IForest 순",
                 Do.assign(d=Do.auroc_tfm - Do.auroc_if).groupby("dataset")[["d", "auroc_tfm", "auroc_if", *DIAGS]].mean().sort_values("d"))
            show(f"M3 진단값 ({tag}): E1 종류별 평균",
                 De.assign(d=De.auroc_tfm - De.auroc_if, type=De.cond.str.removeprefix("type=")).groupby("type")[["d", *DIAGS]].mean()
                 .reindex(S3_TYPES))
            # (a) 진단값 ~ TFM − IForest: 데이터셋 12개 (split 평균) · 36 단위 · E1 46개의 순위 상관, 무너지는 단위 (TFM − IForest ≤ −0.05) 가르기
            dm = Do.assign(d=Do.auroc_tfm - Do.auroc_if).groupby("dataset")[["d", *DIAGS]].mean()
            rows = []
            for d_ in DIAGS:
                if Do[d_].isna().all():
                    continue
                coll = (Do.auroc_tfm - Do.auroc_if <= -0.05).to_numpy()
                rows.append(dict(diag=d_, rho_12=spearmanr(dm[d_], dm.d)[0], rho_36=spearmanr(Do[d_], Do.auroc_tfm - Do.auroc_if)[0],
                                 rho_e1=spearmanr(De[d_], De.auroc_tfm - De.auroc_if)[0],
                                 auc_collapse=roc_auc_score(coll, -Do[d_]) if 0 < coll.sum() < len(coll) else np.nan,
                                 n_collapse=int(coll.sum())))
            show(f"M3 (a) 진단값과 TFM − IForest 의 순위 상관 ({tag}). auc_collapse = 무너지는 단위 (차 ≤ −0.05) 를 낮은 진단값으로 가르는 AUROC",
                 pd.DataFrame(rows).set_index("diag"))
            # (b) 임계값: E1 에서 정해 공식 36 단위에. 주 진단은 방향 고정 (낮으면 IForest), 나머지는 방향도 E1 에서
            mixn = mix_name(tfm, PARTNER, MAIN_RULE, basis)
            cols = {"늘 IForest": PARTNER, f"늘 {tfm}": tfm, "늘 섞기 (주 규칙)": mixn}
            T = So.reindex(Do.index).copy()
            T["_oracle"] = T[[tfm, PARTNER]].max(axis=1)
            rows = []
            for d_ in DIAGS:
                if Do[d_].isna().all() or De[d_].isna().all():
                    continue
                e_hi, e_lo = Se[tfm].reindex(De.index).to_numpy(), Se[PARTNER].reindex(De.index).to_numpy()
                fits = {s: fit_tau(s * De[d_].to_numpy(), e_hi, e_lo) for s in ((1,) if d_.startswith("if") else (1, -1))}
                sign = max(fits, key=lambda s: fits[s][1])
                tau, e1_val = fits[sign]
                o_hi, o_lo = T[tfm].to_numpy(), T[PARTNER].to_numpy()
                sc = pick_scores(Do, T, d_, sign, tau, tfm, PARTNER)
                # 데이터셋 하나씩 빼기: 나머지 11개로 τ (방향은 그대로) → 뺀 데이터셋에
                lodo = pd.Series(np.nan, index=Do.index)
                taus = {}
                for ds in Do.dataset.unique():
                    rest = (Do.dataset != ds).to_numpy()
                    taus[ds], _ = fit_tau(sign * Do[d_].to_numpy()[rest], o_hi[rest], o_lo[rest])
                    lodo[~rest] = pick_scores(Do[~rest], T, d_, sign, taus[ds], tfm, PARTNER)
                tau36, _ = fit_tau(sign * Do[d_].to_numpy(), o_hi, o_lo)
                fit36 = pick_scores(Do, T, d_, sign, tau36, tfm, PARTNER)
                name = pick_name(tfm, PARTNER, d_, basis)
                T[name], T[name + "~lodo"], T[name + "~fit36"] = sc, lodo, fit36
                use = (sign * Do[d_] >= tau).to_numpy()
                rows.append(dict(diag=d_, sign="높으면 TFM" if sign > 0 else "낮으면 TFM", tau_e1=sign * tau, e1_pick=e1_val,
                                 e1_tfm=e_hi.mean(), e1_if=e_lo.mean(), e1_oracle=np.maximum(e_hi, e_lo).mean(),
                                 real_e1tau=by_dataset(sc).mean(), real_lodo=by_dataset(lodo).mean(), real_fit36=by_dataset(fit36).mean(),
                                 tau36=sign * tau36, tfm_units=int(use.sum()),
                                 worst_if=int((~use)[Do.dataset.isin(WORST).to_numpy()].sum()), win_tfm=int(use[Do.dataset.isin(WIN).to_numpy()].sum())))
                if d_ == MAIN_DIAG:
                    cols |= {f"고르기 {name} (τ = E1)": name, "고르기 (데이터셋 하나씩 빼기)": name + "~lodo",
                             "고르기 (36 단위에 맞춘 τ, 상한)": name + "~fit36"}
                    m = (diag.tfm == tfm) & (diag.basis == basis)
                    diag.loc[m, "tau_e1"] = tau
                    diag.loc[m, "pick_e1"] = np.where(diag.loc[m, d_] >= tau, tfm, PARTNER)
                    mo = m & (diag.grid == "official")
                    diag.loc[mo, "tau_lodo"] = diag.loc[mo, "dataset"].map(taus)
                    diag.loc[mo, "pick_lodo"] = np.where(diag.loc[mo, d_] >= diag.loc[mo, "tau_lodo"], tfm, PARTNER)
                    # 변형: 높으면 TFM · 낮으면 섞기, 높으면 섞기 · 낮으면 IForest (τ 는 변형마다 E1 에서)
                    variants = [(tfm, PARTNER, tau)]

                    def vname(hi, lo, d_=d_):  # 섞기 쪽은 mix (그 TFM 의 주 규칙 섞기). 이름에 TFM 이 들어가야 두 TFM 의 행이 구분됨
                        return pick_name(tfm if hi == tfm else f"mix({tfm})", "mix" if lo == mixn else PARTNER, d_, basis)

                    if mixn in Se and mixn in T:
                        for hi, lo in ((tfm, mixn), (mixn, PARTNER)):
                            tv, _ = fit_tau(De[d_].to_numpy(), Se[hi].reindex(De.index).to_numpy(), Se[lo].reindex(De.index).to_numpy())
                            lab = vname(hi, lo)
                            T[lab] = pick_scores(Do, T, d_, 1, tv, hi, lo)
                            cols[f"고르기 {lab} (τ = E1 {tv:.3f})"] = lab
                            variants.append((hi, lo, tv))
                    A = aupr_all["official"].reindex(Do.index)
                    u_ = units("official").set_index("comp")
                    for hi, lo, tv in variants:
                        lab = vname(hi, lo)
                        hi_use = (Do[d_] >= tv).to_numpy()
                        for comp, h in zip(Do.index, hi_use):
                            src = hi if h else lo
                            pick_rows.append(dict(exp="m3", grid="official", comp=comp, dataset=u_.dataset[comp], split=u_.split[comp],
                                                  cond=f"tau={tv:.6f} pick={src}", model=lab, basis=basis, rule="pick", a=hi, b=lo,
                                                  score=T.loc[comp, src], metric_error=1 - T.loc[comp, src], aupr=A.loc[comp, src]))
            show(f"M3 (b) 임계값을 E1 46개에서 정해 공식 36 단위에 ({tag}). real_* = 12개 평균 AUROC, worst_if = 꼴찌 셋 9 단위 중 IForest 를 고른 수, "
                 f"win_tfm = 이기는 넷 12 단위 중 TFM 을 고른 수", pd.DataFrame(rows).set_index("diag"))
            show(f"M3 (b) 주 진단 {MAIN_DIAG} 로 고르기 ({tag})", summary(T, cols | {"상한 (라벨)": "_oracle"}))
            if MAIN_DIAG in Do and not Do[MAIN_DIAG].isna().all():
                name = pick_name(tfm, PARTNER, MAIN_DIAG, basis)
                show(f"M3 데이터셋별 ({tag}): 진단값 (split 평균 · 최소 · 최대), 고른 쪽 (TFM 을 고른 split 수), AUROC",
                     pd.DataFrame({"if10": Do.groupby("dataset")[MAIN_DIAG].mean(), "min": Do.groupby("dataset")[MAIN_DIAG].min(),
                                   "max": Do.groupby("dataset")[MAIN_DIAG].max(),
                                   "tfm_picked": (diag[(diag.tfm == tfm) & (diag.basis == basis) & (diag.grid == "official")]
                                                  .assign(p=lambda d: d.pick_e1 == tfm).groupby("dataset").p.sum()),
                                   "pick": by_dataset(T[name]), tfm: by_dataset(T[tfm]), PARTNER: by_dataset(T[PARTNER])})
                     .reindex([*WIN, "fault", "mammography", "yeast", "Waveform", "shuttle", *WORST]))
    # 타당성 (E2, 배치 기준): test 는 그대로 · context 의 오염만 바뀜 → 진단값이 가림 (라벨로 잰 값) · TFM 의 하락을 따라가는지
    from fab.bench.anomaly.conditions import E2_CELLS

    e2 = diag[(diag.grid == "e2") & (diag.basis == "batch")]
    if len(e2) and E2_CELLS.exists():
        cells = pd.read_csv(E2_CELLS)[["comp", "test_masking", "rate"]]
        rows = []
        for tfm in TFMS:
            g = e2[e2.tfm == tfm].merge(cells, on="comp")
            r0 = g[g.rate == 0].set_index("dataset").auroc_tfm
            g = g[g.rate > 0].assign(drop=lambda d: r0.reindex(d.dataset).to_numpy() - d.auroc_tfm)
            for d_ in ("if10", "rho", "tfm10", "knn10"):
                per = [(spearmanr(x[d_], x.test_masking)[0], spearmanr(x[d_], x["drop"])[0]) for _, x in g.groupby("dataset") if len(x) > 4]
                rows.append(dict(tfm=tfm, diag=d_, datasets=len(per), rho_masking=np.nanmean([p_[0] for p_ in per]),
                                 rho_drop=np.nanmean([p_[1] for p_ in per]),
                                 pooled_masking=spearmanr(g[d_], g.test_masking)[0], pooled_drop=spearmanr(g[d_], g["drop"])[0]))
        show("M3 타당성 (E2 칸, 배치 기준): 진단값과 가림 · TFM 하락의 Spearman (데이터셋 안 상관의 평균 / 88칸 한꺼번에)",
             pd.DataFrame(rows).set_index(["tfm", "diag"]))
    return diag, pick_rows


def report_basis(mix: pd.DataFrame):
    """train 기준의 보정 확인 (라벨 없음): test 행의 분위 평균이 0.5 에서 얼마나 벗어나는지 (방법 · 격자마다, 단위별 평균의 분포)."""
    rows = []
    models = {"official": [PARTNER, "knn", "ecod", MAIN_TFM], "e1": [PARTNER, "knn", "ecod", MAIN_TFM],
              "s3": [PARTNER, MAIN_TFM, *(f"{c}{S3_LA}" for c in CLFS)]}
    for grid, ms in models.items():
        for m in ms:
            v = [q.mean() for u in units(grid).itertuples() if (q := quantiles(u.comp, m, "train")) is not None]
            if v:
                v = np.array(v)
                rows.append(dict(grid=grid, model=m, n=len(v), mean_q=v.mean(), min=v.min(), max=v.max(), mean_abs=np.abs(v - 0.5).mean()))
    if rows:
        show("train 기준 보정 확인: test 행의 분위 평균 (0.5 면 train 점수와 test 점수의 분포가 같음. S3 는 test 에 이상이 4배라 높아야 함)",
             pd.DataFrame(rows).set_index(["grid", "model"]))
    info = []
    for f in sorted(OUTPUTS.glob(f"ad*/*/{TRAIN_INFO}")):
        i = json.loads(f.read_text())
        info.append(dict(grid=grid_of(f.parent.parent.name), model=f.parent.name, kind=i.get("kind"), n=i.get("n"), posinf=i.get("posinf"),
                         nan=i.get("nan"), seconds=i.get("seconds"), refit=i.get("refit_max_abs_diff")))
    if info:
        t = pd.DataFrame(info)
        show("train 행 점수 파일 (대회 수 · 유한하지 않은 행 · 시간 합 (분) · 로컬 다시 fit 한 test 점수의 |차| 최대)",
             t.groupby(["grid", "model", "kind"]).agg(comps=("n", "size"), rows=("n", "sum"), posinf=("posinf", "sum"), nan=("nan", "sum"),
                                                     minutes=("seconds", lambda s: s.sum() / 60), refit_max=("refit", "max")), 4)


def report_cross(diag: pd.DataFrame):
    """교차 절반 (배치 기준만 있는 곳의 민감도): 두 TFM 짝의 공식 · E1 섞기, 공식의 고르기 (다른 절반의 진단값으로 정해 이 절반을 채점)."""
    rows = []
    for grid in ("official", "e1"):
        for tfm in TFMS:
            for rule in ("mean", "max"):
                v = [r for u in units(grid).itertuples() if (r := cross_half(u.comp, tfm, PARTNER, rule, int(u.split[1:])))]
                if v:
                    d = np.array([c - w for c, w in v])
                    rows.append(dict(grid=grid, pair=f"{tfm}&{PARTNER}", rule=rule, n=len(d), cross=np.mean([c for c, _ in v]),
                                     within=np.mean([w for _, w in v]), mean_diff=d.mean(), mean_abs=np.abs(d).mean(), max_abs=np.abs(d).max()))
    if rows:
        show("교차 절반: 다른 절반의 점수 분포를 기준으로 섞은 AUROC − 같은 절반 안의 순위로 섞은 AUROC (단위별)",
             pd.DataFrame(rows).set_index(["grid", "pair", "rule"]), 4)
    rows = []
    for tfm in TFMS:
        d = diag[(diag.tfm == tfm) & (diag.basis == "batch") & (diag.grid == "official")].set_index("comp")
        if d.empty or d.tau_e1.isna().all():
            continue
        tau = float(d.tau_e1.iloc[0])
        cross, within, same = [], [], 0
        for u in units("official").itertuples():
            st, si = load_test(u.comp, tfm), load_test(u.comp, PARTNER)
            h = halves(len(st), int(u.split[1:]))
            for me, other in ((h[0], h[1]), (h[1], h[0])):
                use = {k: diag_values(st[idx], si[idx])[MAIN_DIAG] >= tau for k, idx in (("cross", other), ("within", me))}
                a = {k: grade_rows(u.comp, me, (st if v else si)[me]) for k, v in use.items()}
                if None not in a.values():
                    cross.append(a["cross"])
                    within.append(a["within"])
                    same += use["cross"] == use["within"]
        rows.append(dict(pick=pick_name(tfm, PARTNER, MAIN_DIAG, "batch"), tau_e1=tau, n=len(cross), cross=np.mean(cross),
                         within=np.mean(within), same_choice=same))
    if rows:
        show("교차 절반 (고르기, 공식 36 단위 × 절반 2): 다른 절반의 진단값으로 고른 AUROC 대 같은 절반의 진단값으로 고른 AUROC, 같은 쪽을 고른 수",
             pd.DataFrame(rows).set_index("pick"), 4)


def report_elo(mix: pd.DataFrame, picks: pd.DataFrame, rounds: int):
    """mix 풀: 공식 12 × 3, 기존 10 + 두 TFM + TabPFN 의 섞기 둘 · 고르기 하나 (train 기준) + 같은 셋의 Kumo 판 (@batch). IForest = 1000."""
    from fab.bench.anomaly.conditions import pool_elo
    from fab.bench.anomaly.methods import ANCHOR, PYOD

    sg = singles("official")
    base = sg[sg.model.isin([*PYOD, *TFMS])][["model", "dataset", "split", "score"]]
    want = [mix_name(MAIN_TFM, PARTNER, r, "train") for r in ("mean", "max")] + [pick_name(MAIN_TFM, PARTNER, MAIN_DIAG, "train")] + \
        [mix_name("kumo_unsup", PARTNER, r, "batch") for r in ("mean", "max")] + [pick_name("kumo_unsup", PARTNER, MAIN_DIAG, "batch")]
    both = pd.concat([mix[(mix.grid == "official") & (mix.exp == "m1")], picks], ignore_index=True)
    ent = both[both.model.isin(want)][["model", "dataset", "split", "score"]]
    if miss := sorted(set(want) - set(ent.model)):
        print(f"\nmix 풀: 아직 없는 참가자 {miss} (train 행 점수가 없음) → 빼고 계산")
    long = pd.concat([base, ent], ignore_index=True).assign(metric_error=lambda d: 1 - d.score).rename(columns={"model": "method"})
    t = pool_elo(long, ANCHOR, "mix", rounds)
    if t is None:
        return
    cols = ["pool", "method", "elo", "elo+", "elo-", "rank", "winrate", "imputed", "datasets"]
    t = t[cols].round({"elo": 1, "elo+": 1, "elo-": 1, "rank": 3, "winrate": 4, "imputed": 4})
    t.to_csv(MIX_ELO, index=False)
    print(f"\n== Elo mix (IForest = 1000, 공식 12 × 3 split, 부트스트랩 {rounds}번)\n" + t.drop(columns="pool").to_string(index=False)
          + f"\n{MIX_ELO}: {len(t)}행")


def report(rounds: int):
    from fab.bench.anomaly.methods import LA_LEVELS

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400)
    rows = []
    # M1
    for grid in ("official", "e1"):
        rows += mix_rows("m1", grid, M1_PAIRS, RULES, BASES, cond_text(grid))
    rows += mix_rows("m1", "e2", M1_PAIRS[:2], ["mean", "max"], ["batch"], cond_text("e2"))
    # M2: S3 (la 100%) — 모든 짝은 mean · max, 주 짝은 나머지 규칙도. 대조 = 분류기 & IForest
    s3_pairs = [(f"{c}{S3_LA}", t) for c in CLFS for t in TFMS] + [(f"{MAIN_CLF}{S3_LA}", PARTNER)]
    rows += mix_rows("m2", "s3", s3_pairs, ["mean", "max"], BASES, cond_text("s3"))
    rows += mix_rows("m2", "s3", [(f"{MAIN_CLF}{S3_LA}", MAIN_TFM)], [r for r in RULES if r not in ("mean", "max")], BASES, cond_text("s3"))
    # M2: 공식 × la (배치 기준)
    for k in LA_LEVELS:
        rows += mix_rows("m2", "official", [(f"{c}+la{k}", MAIN_TFM) for c in CLFS], ["mean", "max"], ["batch"], lambda u, g, k=k: f"la={k}")
    mix = pd.DataFrame(rows)
    print(f"섞기 채점 {len(mix)}행: " + ", ".join(f"{k[0]}/{k[1]}/{k[2]} {v}" for k, v in mix.groupby(["exp", "grid", "basis"]).size().items()))
    diag = pd.DataFrame([r for grid in ("official", "e1") for tfm in TFMS for basis in BASES for r in diag_rows(grid, tfm, basis)]
                        + [r for tfm in TFMS for r in diag_rows("e2", tfm, "batch")])

    report_basis(mix)
    report_m1(mix)
    report_m2(mix)
    diag, pick_rows = report_m3(mix, diag)
    report_cross(diag)
    picks = pd.DataFrame(pick_rows)
    report_elo(mix, picks, rounds)

    out = pd.concat([mix, picks], ignore_index=True)
    key = out.assign(cond=out.cond.where(out.exp != "m3", ""))[["exp", "grid", "comp", "cond", "model"]]
    if key.duplicated().any():  # 같은 (단위, 이름) 이 두 번이면 이름이 방법을 못 가름
        raise SystemExit(f"섞기 표에 같은 이름의 행이 겹침: {key[key.duplicated()].model.unique()[:5]}")
    out = out.sort_values(["exp", "grid", "comp", "cond", "model"], kind="stable")
    out.round({"score": 6, "metric_error": 6, "aupr": 6}).to_csv(MIX, index=False)
    diag = diag.sort_values(["grid", "tfm", "basis", "comp"], kind="stable")
    diag.round(6).to_csv(DIAG, index=False)
    print(f"\n{MIX}: {len(out)}행\n{DIAG}: {len(diag)}행")

# --- 명령 ------------------------------------------------------------------------------------------------------


def main(argv: list[str]):
    from fab.bench.anomaly import TRAIN_LABELS

    ap = argparse.ArgumentParser(prog="fab ad mix", description="TFM 을 쓰는 법 (1): 점수 섞기 · 진단으로 고르기 (docs/anomaly.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull", help="Volume → 로컬 (읽기만): 기본은 outputs/ 의 점수 + 내 train 점수 작업의 결과")
    p.add_argument("--data", nargs="*", default=[], choices=TRAIN_GRIDS, help="이 격자의 train.csv · test.csv 도 (로컬 CPU 방법용)")
    p.add_argument("--no-outputs", action="store_true", help="outputs/ 는 안 받음")
    p.add_argument("--no-runs", action="store_true", help="train 점수 작업의 결과는 안 받음")
    p.add_argument("--workers", type=int, default=8)
    sub.add_parser("check", help="받은 test 점수를 다시 채점 → 기존 결과표와 같은지")
    s = sub.add_parser("submit", help="train 행 점수 GPU 작업 (계획만, --yes 로 제출. 정리 작업 큐에 기록하지 않음)")
    s.add_argument("grid", choices=TRAIN_GRIDS)
    s.add_argument("--methods", nargs="+", default=[MAIN_TFM])
    s.add_argument("--yes", action="store_true")
    s.add_argument("--timeout", type=int, default=480, help="작업 시간 제한 (분, 노드 대기 포함)")
    s.add_argument("--accelerator", default=None)
    s.add_argument("--note", default="")
    sub.add_parser("status", help="낸 train 점수 작업의 상태")
    t = sub.add_parser("train", help="로컬 CPU 방법 (pyod · GBDT) 의 train 행 점수")
    t.add_argument("grid", choices=TRAIN_GRIDS)
    t.add_argument("--methods", nargs="+", required=True)
    t.add_argument("--train-labels", default=str(TRAIN_LABELS), help="la 라벨 폴더 (분류기)")
    r = sub.add_parser("report", help="M1 ~ M3 표 · Elo → results/anomaly_mix*.csv")
    r.add_argument("--rounds", type=int, default=200)
    args = ap.parse_args(argv)
    if args.cmd == "pull":
        pull(args.data, not args.no_runs, not args.no_outputs, args.workers)
    elif args.cmd == "check":
        if not check():
            raise SystemExit("기존 결과표와 다름 → 위 목록 확인")
        print("기존 결과표와 같음")
    elif args.cmd == "submit":
        submit(args.grid, args.methods, args.timeout, args.accelerator, args.yes, args.note)
    elif args.cmd == "status":
        status()
    elif args.cmd == "train":
        train_local(args.grid, args.methods, Path(args.train_labels))
    elif args.cmd == "report":
        report(args.rounds)
