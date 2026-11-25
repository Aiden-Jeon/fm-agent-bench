"""TFM (TabPFN 비지도) 이 이득을 보는 조건 (docs/anomaly.md "TFM 이 이득을 보는 조건"). 공식 풀 (anomaly.csv 의 기존 행 ·
anomaly_elo.csv) 은 안 바꿈: 조건 대회 · 변형 방법의 채점은 results/anomaly_conditions.csv, 조건별 Elo 는 anomaly_conditions_elo.csv.

    uv run --frozen fab ad cond traits                 # 데이터 특성 + 이미 있는 채점 → results/anomaly_conditions_traits.csv
    <옆 venv>/python -m fab.cli ad cond prepare e1     # E1 합성 이상 대회 (copulas 가 있는 옆 가상환경, 로컬 CPU)
    uv run --frozen fab ad cond prepare e2             # E2 context 칸 대회
    uv run --frozen fab ad cond submit e0              # 계획만 보여 줌 (대회 · 방법 · 작업). --yes 로 실제 제출 + 채점 요청
    uv run --frozen fab ad cond report                 # 조건별 표 · Elo → results/anomaly_conditions_elo.csv

실험 (번호는 문서와 같음)
    E0  순열 3 (tabpfn_unsup_p3) 대 순열 10: 공식 12개 × s1
    E1  이상 종류: ADBench generate_realistic_synthetic 그대로 (local · global · dependency · cluster) → 대회 adc-e1-<종류>-<이름>-s<seed>
    E2  context 오염 비율 × 모임: s1 의 test 그대로, context = train 정상 전부 + train 이상 n 행 → adc-e2-<이름>-r<비율>-<배치>-d<뽑기>
    E3  라벨 없는 context 정리 (+trim · +iftrim): 공식 12개 × s1
    E4  정상 데이터가 있는 설정: 기존 10개 모두 +clean, 공식 12개 × 3 split (채점 행은 공식 anomaly.csv, 결정 9)
조건 대회의 정답 · train 라벨은 공식 대회와 같은 자리 (FAB_ANOMALY/data, 내 Volume). 모임 정도 (traits) 는 train 라벨로 잰 진단값.
"""

from __future__ import annotations

import ast
import zlib
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from fab.bench.anomaly import (AD_DIR, CONDITIONS, DATASETS, GRADES, SPLITS, adbench_split, latest_grades,
                               load_npz, slug, split_with_ids)
from fab.core.competition import COMP_DIR, Competition, load_competition
from fab.core.results import RESULTS

TRAITS = RESULTS / "anomaly_conditions_traits.csv"
E2_CELLS = RESULTS / "anomaly_conditions_e2_cells.csv"  # E2 칸의 조작 확인 (train 라벨로 잰 진단값)
COND_ELO = RESULTS / "anomaly_conditions_elo.csv"
PREFIX = "adc-"
ADBENCH_GENERATOR = AD_DIR / "src" / "adbench" / "adbench" / "datasets" / "data_generator.py"
# tabpfn-extensions 0.6.3 utils.infer_categorical_features 의 상수 (범주 열 = 값 ≤ 40 개 · 값마다 평균 10행 넘음)
MAX_UNIQUE_CAT, MIN_ROWS_PER_CAT = 40, 10
K = 10  # 이웃 수 (모인 정도)

# 조건 실험의 TabPFN. 결정 10: E0 (순열 3) 가 기준을 못 넘어 (2026-11-25, cover +0.033 · PageBlocks +0.025) 공식과 같은 순열 10
TAB = "tabpfn_unsup"
E0_TAB = "tabpfn_unsup_p3"
KUMO = "kumo_unsup"  # Kumo 로 만든 같은 chain rule (우리 래퍼, methods.kumo_scores)
# E1: ADBench generator 의 기본값 (alpha=5, percentage=0.1). dependency 는 copulas VineCopula 표본이 느려 (로컬 잰 값: 6피처 ·
# 6,666행 6분, 피처 쌍 수 × 행 수에 비례) 36피처 landsat 은 어림 3시간 넘음 → 뺌. PageBlocks 는 copulas 표본에서
# "f(a) and f(b) must have different signs" (scipy brentq, 2026-11-25 로컬) → 뺌 (ADBench 원문 주석도 "copula function may
# occur error in some datasets"). 문서에 적음
E1_TYPES = ["local", "global", "dependency", "cluster"]
E1_ALPHA, E1_PERCENTAGE = 5, 0.1
E1_SKIP = {("dependency", "landsat"), ("dependency", "PageBlocks")}
# E2: 무너짐 셋 · 이김 셋. 비율 r = n / context (%), train 이상의 80% 안에서 되는 칸만. 배치: spread (무작위) · clump (한 이상의 이웃)
E2_DATASETS = ["cover", "cardio", "PageBlocks", "Wilt", "annthyroid", "magic.gamma"]
E2_RATES = [0.5, 1, 2, 5, 10, 20]
E2_ARRANGE = ["spread", "clump"]
E2_DRAWS = [1, 2]
E2_MAX_USE = 0.8
E3_TRIMS = ["+trim5", "+trim10", "+trim20", "+trim10x2", "+iftrim5", "+iftrim10", "+iftrim20"]
E3_MAIN = "+trim10"  # 주 지표 (결정 12: k = 10 고정)


# --- 데이터 특성 (조건 5) ---------------------------------------------------------------------------------------


def categorical_cols(X: np.ndarray) -> list[int]:
    out = []
    for j in range(X.shape[1]):
        u = len(np.unique(X[:, j]))
        if 0 < u <= MAX_UNIQUE_CAT and len(X) / u > MIN_ROWS_PER_CAT:
            out.append(j)
    return out


def neighbour_anomaly_share(X_ref: np.ndarray, y_ref: np.ndarray, X_q: np.ndarray, k: int, self_ref: bool) -> np.ndarray:
    """X_q 의 행마다 X_ref 안 k 이웃 (유클리드, MinMax 공간) 중 이상 비율. self_ref 면 자기 자신을 뺌."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + int(self_ref)).fit(X_ref)
    idx = nn.kneighbors(X_q, return_distance=False)
    if self_ref:
        idx = idx[:, 1:]
    return y_ref[idx].mean(axis=1)


def data_traits(dataset: str, split: str) -> dict:
    X_tr, X_te, y_tr, y_te, _, _ = split_with_ids(dataset, SPLITS[split])
    rate = y_tr.mean()
    cats = categorical_cols(X_tr)
    # 모인 정도: 이상 행의 k 이웃 중 이상 비율 / 이상 비율 (1 = 무작위로 흩어짐, 클수록 모임)
    tr_share = neighbour_anomaly_share(X_tr, y_tr, X_tr[y_tr == 1], K, self_ref=True)
    te_share = neighbour_anomaly_share(X_tr, y_tr, X_te[y_te == 1], K, self_ref=False)  # test 이상 → context 이웃
    te_norm = neighbour_anomaly_share(X_tr, y_tr, X_te[y_te == 0], K, self_ref=False)
    uniq = np.unique(X_tr, axis=0, return_counts=True)[1]
    return dict(dataset=dataset, split=split, n_train=len(y_tr), d=X_tr.shape[1], anomaly_pct=100 * rate,
                n_cat=len(cats), n_cat_multi=sum(len(np.unique(X_tr[:, j])) > 2 for j in cats),
                dup_pct=100 * (uniq[uniq > 1].sum() / len(X_tr)),
                lift_train=tr_share.mean() / rate, lift_test=te_share.mean() / rate,
                share_test_anom=te_share.mean(), share_test_norm=te_norm.mean(),
                # test 이상 중 context 이웃 k 개의 절반 이상이 이상인 비율 (= 자기 무리가 context 에 있음)
                pct_test_anom_in_clump=100 * (te_share >= 0.5).mean())


def outcomes(grades: pd.DataFrame) -> pd.DataFrame:
    """(dataset, split) 마다 이미 있는 채점 (공식 표) 에서 비교에 쓰는 값."""
    from fab.bench.anomaly.methods import CLEAN, PYOD

    g = latest_grades(grades)
    wide = g.pivot_table(index=["dataset", "split"], columns="model", values="score")
    base = wide[[m for m in PYOD if m in wide]]
    out = pd.DataFrame(index=wide.index)
    out["tabpfn"] = wide.get("tabpfn_unsup")
    out["tabpfn_clean"] = wide.get("tabpfn_unsup" + CLEAN)
    out["iforest"] = wide.get("iforest")
    out["iforest_clean"] = wide.get("iforest" + CLEAN)
    out["best_base"] = base.max(axis=1)
    out["best_base_method"] = base.idxmax(axis=1)
    out["median_base"] = base.median(axis=1)
    out["tabpfn_minus_best"] = out.tabpfn - out.best_base
    out["tabpfn_minus_iforest"] = out.tabpfn - out.iforest
    out["clean_gain"] = out.tabpfn_clean - out.tabpfn
    out["tabpfn_rank"] = pd.concat([base, out.tabpfn.rename("tabpfn")], axis=1).rank(axis=1, ascending=False)["tabpfn"]
    return out.reset_index()


def traits_table() -> pd.DataFrame:
    t = pd.DataFrame([data_traits(ds, s) for ds in DATASETS for s in SPLITS])
    res = t.merge(outcomes(pd.read_csv(GRADES)), on=["dataset", "split"], how="left")
    res.round(4).to_csv(TRAITS, index=False)
    print(f"{TRAITS}: {len(res)}행")
    return res


def spearman_summary(res: pd.DataFrame) -> pd.DataFrame:
    """데이터셋 단위 (split 평균) 순위 상관: 특성 × 결과."""
    from scipy.stats import spearmanr

    feats = ["anomaly_pct", "d", "n_train", "n_cat", "dup_pct", "lift_train", "lift_test", "share_test_anom",
             "pct_test_anom_in_clump", "median_base", "best_base"]
    targets = ["tabpfn_minus_best", "tabpfn_minus_iforest", "clean_gain"]
    m = res.groupby("dataset")[feats + targets].mean(numeric_only=True)
    rows = []
    for f in feats:
        r = dict(trait=f)
        for t in targets:
            rho, p = spearmanr(m[f], m[t])
            r[t], r[f"{t}_p"] = rho, p
        rows.append(r)
    return pd.DataFrame(rows)


# --- 조건 대회 ---------------------------------------------------------------------------------------------------


@cache
def adbench_synthetic():
    """ADBench data_generator.py 의 DataGenerator.generate_realistic_synthetic 를 원문 그대로 (소스에서 그 함수만 떼어 실행).
    모듈째 import 하면 adbench.myutils 가 tensorflow 를 부름 → 함수 하나만. 쓰는 이름 (np · pd · GaussianMixture · VineCopula ·
    GaussianKDE) 은 원문 import 와 같은 것. copulas 는 dependency 에서만 쓰므로 없으면 그 종류만 실패."""
    from sklearn.mixture import GaussianMixture

    ns = dict(np=np, pd=pd, GaussianMixture=GaussianMixture)
    try:
        from copulas.multivariate import VineCopula
        from copulas.univariate import GaussianKDE

        ns |= dict(VineCopula=VineCopula, GaussianKDE=GaussianKDE)
    except ImportError:
        pass
    tree = ast.parse(ADBENCH_GENERATOR.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataGenerator")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate_realistic_synthetic")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ADBENCH_GENERATOR), "exec"), ns)  # noqa: S102 - ADBench 원문
    return ns["generate_realistic_synthetic"]


def comp_ids(dataset: str, seed: int, tag: str, n: int) -> np.ndarray:
    num = int(DATASETS[dataset].split("_", 1)[0])
    return np.random.default_rng([num, seed, zlib.crc32(tag.encode())]).permutation(n)


def e1_split(dataset: str, mode: str, seed: int):
    """ADBench generator (realistic_synthetic_mode=mode, la=0) 와 같은 순서: set_seed → 1만 행 표본 → 합성 → 층화 70/30 → MinMax."""
    if mode == "dependency":
        try:
            import copulas  # noqa: F401
        except ImportError:
            raise SystemExit("dependency 는 copulas 가 필요: 옆 가상환경 (FAB_ANOMALY/.venv) 의 python 으로 prepare") from None
    fn = adbench_synthetic()
    X, y = load_npz(dataset)

    def synth(X, y):
        return fn(SimpleNamespace(seed=seed), X, y, realistic_synthetic_mode=mode, alpha=E1_ALPHA,
                  percentage=E1_PERCENTAGE)

    X_tr, X_te, y_tr, y_te = adbench_split(X, y, seed, synth=synth)
    ids = comp_ids(dataset, seed, f"e1-{mode}", len(y_tr) + len(y_te))
    return X_tr, X_te, y_tr, y_te, ids[:len(y_tr)], ids[len(y_tr):]


def e2_count(n_normal: int, rate: float) -> int:
    return int(round(rate / (100 - rate) * n_normal))


def e2_cells(dataset: str) -> list[dict]:
    """(비율, 배치, 뽑기) 칸. r = 0 은 하나 (= 정상만, clean 과 같은 context)."""
    y = split_with_ids(dataset, SPLITS["s1"])[2]
    n_norm, n_anom = int((y == 0).sum()), int(y.sum())
    cells = [dict(rate=0, arrange="none", draw=0, n_anomalies=0)]
    for r in E2_RATES:
        if 0 < (n := e2_count(n_norm, r)) <= E2_MAX_USE * n_anom:
            cells += [dict(rate=r, arrange=a, draw=d, n_anomalies=n) for a in E2_ARRANGE for d in E2_DRAWS]
    return cells


def e2_split(dataset: str, cond: dict):
    """s1 분할 그대로 (test · 척도 · 행 id 같음 → 정답이 s1 과 같음). context 만: train 정상 전부 + train 이상 n 행 (행 순서 그대로)."""
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_with_ids(dataset, SPLITS["s1"])
    anom = np.flatnonzero(y_tr == 1)
    n = cond["n_anomalies"]
    rng = np.random.default_rng([int(DATASETS[dataset].split("_", 1)[0]), int(cond["draw"]), 2])
    if n == 0:
        pick = anom[:0]
    elif cond["arrange"] == "spread":
        pick = rng.choice(anom, n, replace=False)
    elif cond["arrange"] == "clump":  # 무작위 이상 하나를 중심으로 train 이상 중 가장 가까운 n 개 (중심 포함)
        center = X_tr[rng.choice(anom)]
        pick = anom[np.argsort(np.linalg.norm(X_tr[anom] - center, axis=1), kind="stable")[:n]]
    else:
        raise ValueError(cond["arrange"])
    keep = (y_tr == 0) | np.isin(np.arange(len(y_tr)), pick)
    return X_tr[keep], X_te, y_tr[keep], y_te, id_tr[keep], id_te


def e2_cells_table() -> pd.DataFrame:
    """E2 칸마다 조작 확인: context 이상끼리의 평균 거리 (작을수록 모임) 와 test 이상의 가림 (context 이웃 10개 중 이상 비율 =
    test 이상과 닮은 이상이 context 에 있는 정도). 모임 배치는 거리를 줄이지만 한 곳에만 있어 가림은 줄 수 있음 → 둘 다 남김."""
    from scipy.spatial.distance import pdist

    rows = []
    for c in e2_comps(E2_DATASETS):
        ad = c.anomaly
        X_tr, X_te, y_tr, y_te, _, _ = e2_split(ad["dataset"], ad["condition"])
        a = X_tr[y_tr == 1]
        rows.append(dict(comp=c.name, dataset=ad["dataset"], **{k: v for k, v in ad["condition"].items() if k != "exp"},
                         context_rows=len(y_tr),
                         anomaly_pairdist=float(pdist(a).mean()) if len(a) > 1 else np.nan,
                         test_masking=float(neighbour_anomaly_share(X_tr, y_tr, X_te[y_te == 1], K, False).mean())))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(E2_CELLS, index=False)
    print(f"{E2_CELLS}: {len(out)}행")
    return out


def condition_split(comp: Competition):
    """prepare (fab/bench/anomaly) 가 부름: 조건 대회의 (X_train, X_test, y_train, y_test, id_train, id_test)."""
    ad, c = comp.anomaly, comp.anomaly["condition"]
    if c["exp"] == "e1":
        return e1_split(ad["dataset"], c["type"], int(ad["seed"]))
    if c["exp"] == "e2":
        return e2_split(ad["dataset"], c)
    raise ValueError(f"알 수 없는 조건 실험: {c}")


def rate_tag(r: float) -> str:
    return f"{r:g}".replace(".", "p")


def cond_comp(name: str, dataset: str, seed: int, condition: dict) -> Competition:
    """competitions/adc-*/config.yaml (git 제외). 대회 설명은 없음 (Agent 를 안 돌림)."""
    cfg = dict(name=name, anomaly=dict(dataset=dataset, adbench=DATASETS[dataset], split=f"s{seed}", seed=seed,
                                       condition=condition),
               task="anomaly", target="label", id_col="id", metric="auroc", cv={})
    d = COMP_DIR / name
    d.mkdir(exist_ok=True)
    text = "# 생성 파일 (fab/bench/anomaly/conditions.py, git 제외). 조건 실험 대회\n" + \
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if not (d / "config.yaml").exists() or (d / "config.yaml").read_text() != text:
        (d / "config.yaml").write_text(text)
    return load_competition(name)


def e1_comps(types: list[str], datasets: list[str], seeds: list[int]) -> list[Competition]:
    return [cond_comp(f"{PREFIX}e1-{t}-{slug(ds)}-s{s}", ds, s, dict(exp="e1", type=t))
            for t in types for ds in datasets for s in seeds if (t, ds) not in E1_SKIP]


def e2_comps(datasets: list[str]) -> list[Competition]:
    out = []
    for ds in datasets:
        for c in e2_cells(ds):
            tail = "r0" if c["rate"] == 0 else f"r{rate_tag(c['rate'])}-{c['arrange']}-d{c['draw']}"
            out.append(cond_comp(f"{PREFIX}e2-{slug(ds)}-{tail}", ds, 1, dict(exp="e2", **c)))
    return out


def official(splits: list[str]) -> list[Competition]:
    from fab.bench.anomaly import make_comps

    return [c for ds in DATASETS for c in make_comps(ds, splits)]


def _prepare_one(name: str) -> str:
    from fab.bench.anomaly import ensure_prepared

    ensure_prepared(load_competition(name))
    return name


def prepare_all(comps: list[Competition], workers: int):
    """대회마다 prepare (이미 있으면 건너뜀). 합성 (특히 dependency) 이 느려 프로세스 여럿으로."""
    names = [c.name for c in comps]
    if workers <= 1:
        for n in names:
            _prepare_one(n)
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        for n in ex.map(_prepare_one, names):
            print(f"  준비 끝: {n}", flush=True)


# --- 실험 정의 · 제출 --------------------------------------------------------------------------------------------


def plan(exp: str, tab: str, types: list[str], datasets: list[str] | None, only: str | None = None,
         keep: list[str] | None = None) -> tuple[list[Competition], list[str], str]:
    """only = cpu · gpu 면 그쪽 방법만 (pyod 부분은 E0 를 기다릴 필요가 없음). keep: 이 방법들만 (예: Kumo 의 E3 는 trim10 · iftrim10)."""
    from fab.bench.anomaly.methods import is_gpu

    comps, methods, label = _plan(exp, tab, types, datasets)
    if only:
        methods = [m for m in methods if is_gpu(m) == (only == "gpu")]
        label += f"-{only}"
    if keep:
        if bad := sorted(set(keep) - set(methods)):
            raise SystemExit(f"{exp} 계획에 없는 방법: {bad} (있는 것: {' '.join(methods)})")
        methods = [m for m in methods if m in keep]
    if tab != TAB:
        label += f"-{tab.split('_')[0]}"
    return comps, methods, label


def _plan(exp: str, tab: str, types: list[str], datasets: list[str] | None) -> tuple[list[Competition], list[str], str]:
    """실험 → (대회, 방법, 작업 이름)."""
    from fab.bench.anomaly.methods import ANCHOR, CLEAN, PYOD

    pyod = list(PYOD)
    if exp == "e0":
        return official(["s1"]), [E0_TAB], "adc-e0"
    if exp == "e1":
        comps = e1_comps(types, datasets or list(DATASETS), [1])
        return comps, [*pyod, tab, tab + CLEAN, *(p + CLEAN for p in pyod)], "adc-e1"
    if exp == "e2":
        return e2_comps(datasets or E2_DATASETS), [*pyod, tab], "adc-e2"
    if exp == "e3":
        return official(["s1"]), [*(tab + t for t in E3_TRIMS), *(p + E3_MAIN for p in pyod)], "adc-e3"
    if exp == "e4":
        return official(list(SPLITS)), [p + CLEAN for p in pyod if p != ANCHOR], "adc-e4"
    raise SystemExit(f"실험은 e0 ~ e4: {exp}")


def submit(exp: str, tab: str, types: list[str], datasets: list[str] | None, timeout: int, accelerator: str | None,
           yes: bool, note: str, only: str | None, keep: list[str] | None = None):
    from fab.bench.anomaly import submit_comps
    from fab.bench.anomaly.methods import is_gpu

    comps, methods, label = plan(exp, tab, types, datasets, only, keep)
    if not methods:
        raise SystemExit(f"{exp} --only {only}: 낼 방법이 없음")
    gpu = [m for m in methods if is_gpu(m)]
    cpu = [m for m in methods if not is_gpu(m)]
    print(f"{exp}: 대회 {len(comps)}개 (예: {', '.join(c.name for c in comps[:3])} …)\n"
          f"  GPU 작업 방법 {len(gpu)}개: {' '.join(gpu) or '-'}\n  CPU 작업 방법 {len(cpu)}개: {' '.join(cpu) or '-'}")
    missing = [c.name for c in comps if not (c.data_dir / "train.csv").exists()]
    if missing:
        raise SystemExit(f"준비 안 된 대회 {len(missing)}개 (예: {missing[:3]}) → fab ad cond prepare {exp}")
    if not yes:
        print("계획만 (제출은 --yes)")
        return
    submit_comps(comps, methods, note or label, timeout, accelerator, True, label)
    print("끝난 뒤 fab remote kick (이 worktree 에서) → fab sync → fab ad cond report")


# --- 결과 · Elo --------------------------------------------------------------------------------------------------


def parse_cond(s) -> dict:
    return dict(kv.split("=", 1) for kv in str(s).split()) if isinstance(s, str) and s else {}


def read_conditions() -> pd.DataFrame:
    if not CONDITIONS.exists():
        return pd.DataFrame(columns=["comp", "model", "dataset", "split", "score", "metric_error", "exp", "cond"])
    g = latest_grades(pd.read_csv(CONDITIONS))
    c = g.cond.map(parse_cond)
    for k in ("type", "rate", "arrange", "draw"):
        g[k] = c.map(lambda d, k=k: d.get(k))
    return g


def fill_trim(long: pd.DataFrame) -> pd.DataFrame:
    """정리 변형 (+trim · +iftrim) 이 실패한 칸 = 정리하지 않은 같은 방법의 값 (imputed = 1). 실패는 정리 뒤 context 에서 조건 열이
    상수가 되어 TabPFN 이 거부한 것 (cardio · yeast, 2026-11-26). 기준점 (IForest) 값으로 채우면 TabPFN 이 크게 지는 cardio 에서
    정리 변형이 IForest 값을 받아 부풀어 오름 → "정리에 실패하면 정리 안 함" 으로 채움."""
    from fab.bench.anomaly.methods import base as base_name

    long = long.assign(imputed=0.0)
    keys = long[["dataset", "split"]].drop_duplicates()
    add = []
    for m in [m for m in long.method.unique() if "trim" in m]:
        have = long[long.method == m][["dataset", "split"]]
        miss = keys.merge(have, how="left", indicator=True).query("_merge == 'left_only'")[["dataset", "split"]]
        src = long[long.method == base_name(m)].merge(miss, on=["dataset", "split"])
        add.append(src.assign(method=m, imputed=1.0))
    return pd.concat([long, *add], ignore_index=True)


def pool_elo(long: pd.DataFrame, anchor: str, name: str, rounds: int) -> pd.DataFrame | None:
    from fab.bench.tabarena import elo

    long = long[["method", "dataset", "split", "metric_error", *(["imputed"] if "imputed" in long else [])]]
    long = long.dropna(subset=["metric_error"])
    if long.empty or anchor not in set(long.method):
        print(f"{name}: 기준점 {anchor} 가 없음 → 건너뜀")
        return None
    filled = elo.fill(long, list(dict.fromkeys(long.method)), anchor=anchor)
    t = elo.elo(filled, rounds=rounds, anchor=anchor)
    return t.reset_index().assign(pool=name)


def report(rounds: int, tab: str):
    from fab.bench.anomaly.methods import ANCHOR, CLEAN, OFFICIAL, PYOD

    off = latest_grades(pd.read_csv(GRADES))
    cond = read_conditions()
    pools = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    # E0: 순열 3 대 10 (s1)
    e0 = cond[(cond.exp == "base") & (cond.model == E0_TAB) & (cond.split == "s1")].set_index("dataset").score
    if len(e0):
        ref = off[(off.model == "tabpfn_unsup") & (off.split == "s1")].set_index("dataset").score
        t = pd.DataFrame({"p10": ref, E0_TAB: e0}).dropna()
        t["diff"] = t[E0_TAB] - t.p10
        print(f"\n== E0: {E0_TAB} 대 tabpfn_unsup (s1)\n{t.round(4).to_string()}\n"
              f"|차| ≤ 0.01: {int((t['diff'].abs() <= 0.01).sum())}/{len(t)}, 최대 {t['diff'].abs().max():.4f}")

    # E4: clean 풀 (공식 대회 12 × 3, 기존 10 + TabPFN 모두 +clean, IForest+clean = 1000)
    cl = off[off.model.str.endswith(CLEAN) & off.model.str.removesuffix(CLEAN).isin(OFFICIAL)]
    if cl.model.nunique() > 2:
        pools.append(pool_elo(cl.rename(columns={"model": "method"}), ANCHOR + CLEAN, "clean", rounds))

    # E1: 종류마다 비지도 풀 (기존 10 + TabPFN, IForest = 1000) · 종류별 평균 표
    e1 = cond[cond.exp == "e1"]
    for t in E1_TYPES:
        g = e1[(e1.type == t) & ~e1.model.str.endswith(CLEAN)]
        if len(g):
            pools.append(pool_elo(g.rename(columns={"model": "method"}), ANCHOR, f"e1:{t}", rounds))
    if len(e1):
        w = e1.pivot_table(index=["type", "dataset"], columns="model", values="score")
        print("\n== E1: 종류별 평균 AUROC (데이터셋 평균)\n" + w.groupby("type").mean().T.round(3).to_string())

    # E2: (데이터셋, 비율, 배치) 마다 뽑기 평균
    e2 = cond[cond.exp == "e2"]
    if len(e2):
        e2 = e2.assign(rate=e2.rate.astype(float))
        w = e2.pivot_table(index=["dataset", "rate", "arrange"], columns="model", values="score", aggfunc="mean")
        cols = [c for c in [tab, ANCHOR, "knn", "lof", "ecod", "loda"] if c in w]
        print("\n== E2: 뽑기 평균 AUROC\n" + w[cols].round(3).to_string())
        if E2_CELLS.exists():  # 칸마다 r = 0 대비 떨어진 몫 대 조작 측정값 (가림 · 모인 정도)
            from scipy.stats import spearmanr

            cells = pd.read_csv(E2_CELLS)[["comp", "test_masking", "anomaly_pairdist"]]
            sc = e2.pivot_table(index=["comp", "dataset", "rate", "arrange"], columns="model", values="score").reset_index()
            r0 = sc[sc.rate == 0].set_index("dataset")
            sc = sc[sc.rate > 0].merge(cells, on="comp")
            rows = []
            for m in [c for c in cols if c in sc]:
                drop = r0[m].reindex(sc.dataset).to_numpy() - sc[m].to_numpy()
                ok = ~np.isnan(drop)
                rows.append(dict(model=m, rho_drop_masking=spearmanr(drop[ok], sc.test_masking[ok])[0],
                                 rho_drop_rate=spearmanr(drop[ok], sc.rate[ok])[0],
                                 drop_clump=np.nanmean(drop[(sc.arrange == "clump").to_numpy()]),
                                 drop_spread=np.nanmean(drop[(sc.arrange == "spread").to_numpy()]), n=int(ok.sum())))
            print("\n== E2: r = 0 대비 떨어진 AUROC (칸 단위, Spearman)\n" + pd.DataFrame(rows).round(3).to_string(index=False))

    # E3: trim 풀 (공식 12 × s1, 기존 10 + TabPFN (순열 같게) + 정리 변형)
    # 같은 순열의 TabPFN 기준 (순열 10 = 공식 tabpfn_unsup 의 s1, 그 밖은 조건 표의 것) + 정리 변형 (+trim · +iftrim)
    e3 = cond[(cond.exp == "base") & (cond.split == "s1") & (cond.model.str.contains("trim") | (cond.model == tab))]
    e3 = e3[~e3.model.str.contains("trim") | e3.model.str.startswith(tab + "+") | ~e3.model.str.startswith("tabpfn")]
    if len(e3):
        base = off[off.model.isin([*PYOD, *([tab] if tab in OFFICIAL else [])]) & (off.split == "s1")]
        g = pd.concat([base, e3], ignore_index=True).rename(columns={"model": "method"})
        pools.append(pool_elo(fill_trim(g), ANCHOR, "trim", rounds))
        w = g.pivot_table(index="dataset", columns="method", values="score")
        print("\n== E3: AUROC (s1)\n" + w.round(3).to_string())
        # 메운 몫 = (정리 − 오염) / (clean − 오염): clean 진단 (train 라벨) 과의 차이 중 라벨 없이 메운 비율
        cl = off[(off.model == tab + CLEAN) & (off.split == "s1")].set_index("dataset").score
        if len(cl) and tab in w:
            gap = (cl - w[tab]).rename("clean_gap")
            fr = pd.concat([w[tab].rename("contaminated"), cl.rename("clean"), gap,
                            *[((w[c] - w[tab]) / gap).rename(c.removeprefix(tab)) for c in w if c.startswith(tab + "+")]], axis=1)
            print("\n== E3: 메운 몫 (정리 − 오염) / (clean − 오염), 빈칸 = 정리 실패\n" + fr.round(2).to_string()
                  + "\n중앙값\n" + fr.median().round(2).to_string())

    # Kumo (우리 래퍼): 공식 대회의 kumo_unsup (조건 표) 와 기존 10 + tabpfn_unsup (공식 표) 을 같은 (데이터셋, split) 에서. +clean 은 표만
    ku = cond[(cond.exp == "base") & cond.model.isin([KUMO, KUMO + CLEAN])]
    if len(ku):
        keys = set(zip(ku.dataset, ku.split))
        o = off[off.model.isin([*OFFICIAL, "tabpfn_unsup" + CLEAN]) & pd.Series(list(zip(off.dataset, off.split)),
                                                                              index=off.index).isin(keys)]
        g = pd.concat([o, ku], ignore_index=True)
        pools.append(pool_elo(g[~g.model.str.endswith(CLEAN)].rename(columns={"model": "method"}), ANCHOR, "kumo", rounds))
        w = g.pivot_table(index="dataset", columns="model", values="score")
        cols = [c for c in ["tabpfn_unsup", KUMO, "tabpfn_unsup" + CLEAN, KUMO + CLEAN, ANCHOR] if c in w]
        print("\n== Kumo (우리 래퍼) 대 TabPFN, split 평균 AUROC\n" + w[cols].round(3).to_string())

    pools = [p for p in pools if p is not None]
    if pools:
        res = pd.concat(pools, ignore_index=True)
        cols = ["pool", "method", "elo", "elo+", "elo-", "rank", "winrate", "imputed", "datasets"]
        res = res[cols]
        res.round({"elo": 1, "elo+": 1, "elo-": 1, "rank": 3, "winrate": 4, "imputed": 4}).to_csv(COND_ELO, index=False)
        for p, t in res.groupby("pool", sort=False):
            print(f"\n== Elo {p}\n" + t.drop(columns="pool").round(1).to_string(index=False))
        print(f"\n{COND_ELO}: {len(res)}행")


def main(argv: list[str]):
    import argparse

    ap = argparse.ArgumentParser(prog="fab ad cond", description="TFM 이 이득을 보는 조건 (docs/anomaly.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("traits", help="데이터 특성 + 이미 있는 채점 → results/anomaly_conditions_traits.csv")
    for cmd in ("prepare", "submit"):
        p = sub.add_parser(cmd, help="조건 대회 준비 (로컬)" if cmd == "prepare" else "계획 보기 · --yes 로 제출 + 채점 요청")
        p.add_argument("exp", choices=["e0", "e1", "e2", "e3", "e4"])
        p.add_argument("--types", nargs="+", default=E1_TYPES, choices=E1_TYPES, help="E1 의 이상 종류")
        p.add_argument("--datasets", nargs="+", default=None)
        p.add_argument("--tabpfn", default=TAB, help=f"조건 실험의 FM (기본 {TAB}, Kumo 판은 {KUMO})")
        if cmd == "prepare":
            p.add_argument("--workers", type=int, default=1)
        else:
            p.add_argument("--yes", action="store_true")
            p.add_argument("--only", choices=["cpu", "gpu"], default=None, help="그쪽 작업만 (기본: 둘 다)")
            p.add_argument("--methods", nargs="+", default=None, help="계획 중 이 방법들만")
            p.add_argument("--timeout", type=int, default=360, help="작업 시간 제한 (분)")
            p.add_argument("--accelerator", default=None)
            p.add_argument("--note", default="")
    r = sub.add_parser("report", help="조건별 표 · Elo → results/anomaly_conditions_elo.csv")
    r.add_argument("--rounds", type=int, default=200)
    r.add_argument("--tabpfn", default=TAB)
    args = ap.parse_args(argv)
    if getattr(args, "datasets", None):
        from fab.bench.anomaly import resolve

        args.datasets = [resolve(d) for d in args.datasets]
    if args.cmd == "traits":
        res = traits_table()
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            m = res.groupby("dataset").mean(numeric_only=True).sort_values("tabpfn_minus_best")
            print(m.round(3).to_string())
            print("\n순위 상관 (데이터셋 12개, split 평균)")
            print(spearman_summary(res).round(3).to_string(index=False))
    elif args.cmd == "prepare":
        comps, _, _ = plan(args.exp, args.tabpfn, args.types, args.datasets)
        prepare_all(comps, args.workers)
        print(f"{args.exp}: 대회 {len(comps)}개 준비")
        if args.exp == "e2":
            e2_cells_table()
    elif args.cmd == "submit":
        submit(args.exp, args.tabpfn, args.types, args.datasets, args.timeout, args.accelerator, args.yes, args.note,
               args.only, args.methods)
    elif args.cmd == "report":
        report(args.rounds, args.tabpfn)
