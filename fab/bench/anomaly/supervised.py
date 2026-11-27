"""라벨이 있는 이상탐지 (지도 · 반지도) 와 FoMo-0D 기준점 (docs/anomaly.md "라벨이 있는 이상탐지"). 공식 비지도 행 (anomaly.csv 의 기존
행) 은 안 바꿈: 분류기 (+la<k>) 와 S3 대회의 채점은 results/anomaly_supervised.csv, 그 Elo 는 anomaly_supervised_elo.csv.

    uv run --frozen fab ad sup prepare s12          # 공식 12개 × 3 split 의 la 라벨 파일 (ADBench 와 같이 고름) → FAB_ANOMALY/data/train_labels/<대회>.la.csv
    uv run --no-project --python $FAB_ANOMALY/.venv/bin/python python -m fab ad sup prepare s3   # S3 대회 (copulas 가 있는 옆 가상환경)
    uv run --frozen fab ad sup submit s12           # 계획만 (대회 · 방법 · 작업). --yes 로 제출 + 채점 요청
    uv run --frozen fab ad sup report               # 표 · Elo → results/anomaly_supervised_elo.csv

실험 (번호는 문서와 같음)
    S1 · S2  분류기 4개 × la ∈ {1, 5, 10, 25, 50, 100}%: 공식 12개 × 3 split (계획 이름 s12)
    S3       처음 보는 이상: 정상 하나 (GMM) + 네 종류 이상, train 종류 A 마다 대회 adc-s3-<A>-<이름>-s<seed>, test 는 넷이 같음
             (정답의 group 열 = test 이상의 종류 → 채점이 종류 B 마다 한 행)
    F0 ~ F2  FoMo-0D: 공식 12개 × 3 split (fomo0d · +clean, 공식 풀), E1 46개 (fomo0d · +clean), E2 94칸 (fomo0d)
    L1       라벨 준 이상을 context 에서 뺌 (+dropla<k>): 공식 12개 × s1, TabPFN · Kumo 비지도 + 대조 IForest · KNN · LOF (docs/anomaly.md
             "라벨이 조금 있을 때 (2)")
    L2       처음 보는 종류 + 라벨 조금: S3 40 대회의 분류기를 la 10% (주) · 1% 로
    L3       tabpfn_unsup 의 S3 fault 두 대회 (점수 +inf 를 맨 위 유한 점수로, methods.top_posinf)
"""

from __future__ import annotations

import copy
import math
import random
from functools import cache

import numpy as np
import pandas as pd

from fab.bench.anomaly import (DATASETS, GRADES, ID_COL, LABEL_COL, MAX_ROWS, SPLITS, SUPERVISED, TEST_SIZE,
                               TRAIN_LABELS, latest_grades, load_npz, slug, split_with_ids)
from fab.core.competition import Competition
from fab.core.results import RESULTS

SUP_ELO = RESULTS / "anomaly_supervised_elo.csv"
REFERENCE_SUP = RESULTS / "anomaly_reference_supervised.csv"  # ADBench Table D6 ~ D18 (AUROC) 의 우리 12개 행
CLFS = ["tabpfn_clf", "kumo_clf", "lgbm_clf", "catboost_clf"]
# 비지도 참고 (같은 split 의 이미 있는 채점): 공식 표의 기존 10 + tabpfn_unsup + fomo0d, 조건 표의 kumo_unsup
UNSUP_REF = ["iforest", "knn", "tabpfn_unsup", "kumo_unsup", "fomo0d"]
S3_TYPES = ["local", "global", "dependency", "cluster"]
S3_SEED = 1
S3_CPU = ["iforest", "knn", "lof"]
S3_GPU_UNSUP = ["tabpfn_unsup", "kumo_unsup", "fomo0d"]  # kumo_unsup: 2026-11-26 사용자 (결정 20)
# L1 (결정 26 ~ 28, 2026-11-27 사용자): s1 × la {1, 5, 10, 25, 50, 75}. Kumo 는 la 세 값만 (E2 의 Kumo 판이 TabPFN 과 같은 모양)
L1_SPLIT = "s1"
L1_LA = [1, 5, 10, 25, 50, 75]
L1_KUMO_LA = [10, 50, 75]
L1_TFM = ["tabpfn_unsup", "kumo_unsup"]
L1_CONTROL = ["iforest", "knn", "lof"]  # 같은 빼기를 기존 방법에 (TFM 만의 효과인지. KNN · LOF 는 E4 에서 TFM 만큼 오염에 다침)
L1_FILL_MIN = 0.5  # "살아난다" = clean 과의 차이를 절반 이상 메움 (H12a, E3 의 H3 과 같은 기준)
L1_COLLAPSED = ["cover", "cardio", "PageBlocks"]  # 오염으로 무너지는 셋 (보조 지표: 오염 그대로의 IForest 를 넘는 가장 작은 la)
# L2 (결정 29): la 10% 가 판정용 (H13), la 1% 는 민감도. S3 대회의 la 열 = 고정 seed 순열의 앞 ceil(la × A 이상 수) 개
L2_LA = [10, 1]
L2_MAIN = 10
S3_LA = [1, 10, 100]
# L3 (결정 30): tabpfn_unsup 이 점수 +inf 로 실패했던 S3 대회 (fault 의 train = local · cluster)
L3_COMPS = [("local", "fault"), ("cluster", "fault")]


# --- la 라벨 (ADBench generator 의 at_least_one_labeled=True 와 같은 고르기) -------------------------------------


def adbench_la_labels(dataset: str, seed: int, levels: list[int]) -> pd.DataFrame:
    """공식 분할 (split_with_ids = adbench_split + id) 바로 뒤의 numpy 전역 상태에서 ADBench 처럼 la 마다 새로
    `np.random.choice(idx_anomaly, ceil(la × n), replace=False)` → id + la<k> 열 (라벨 준 이상 1, 나머지 0 = PU).
    split_with_ids 는 adbench_split 뒤에 np.random.default_rng (전역 상태와 따로) 만 쓰고, MinMax 는 난수를 안 씀 → 상태가 ADBench
    의 그 자리와 같음. la 마다 같은 상태에서 고르므로 (numpy 비복원 choice = permutation[:k]) 작은 la 의 집합이 큰 la 에 포함됨."""
    _, _, y_tr, _, id_tr, _ = split_with_ids(dataset, seed)
    state = np.random.get_state()
    idx = np.flatnonzero(y_tr == 1)
    out = pd.DataFrame({ID_COL: id_tr})
    for k in levels:
        np.random.set_state(state)
        pick = np.random.choice(idx, math.ceil(k / 100 * len(idx)), replace=False)
        lab = np.zeros(len(y_tr), dtype=int)
        lab[pick] = 1
        out[f"la{k}"] = lab
    np.random.set_state(state)
    return out


def s3_la_labels(comp: Competition, levels: list[int]) -> pd.DataFrame:
    """S3 대회의 la 라벨. S3 에는 ADBench 분할이 없어 "분할 바로 뒤의 전역 상태" 가 없음 → 대회마다 고정 seed 의 순열 하나에서 앞
    ceil(la × A 이상 수) 개 (개수 규칙은 ADBench 와 같고, 작은 la 의 집합이 큰 la 에 포함됨). la100 = train 의 A 이상 모두."""
    ad = comp.anomaly
    full = pd.read_csv(TRAIN_LABELS / f"{comp.name}.csv")
    idx = np.flatnonzero(full[LABEL_COL].to_numpy() == 1)
    num = int(DATASETS[ad["dataset"]].split("_", 1)[0])
    perm = np.random.default_rng([num, int(ad["seed"]), 60 + S3_TYPES.index(ad["condition"]["train"])]).permutation(idx)
    out = pd.DataFrame({ID_COL: full[ID_COL]})
    for k in levels:
        lab = np.zeros(len(full), dtype=int)
        lab[perm[: math.ceil(k / 100 * len(idx))]] = 1
        out[f"la{k}"] = lab
    return out


def write_la_labels(comp: Competition):
    """<대회>.la.csv (id + la<k> 열). 공식 대회는 ADBench 와 같은 고르기 (adbench_la_labels) 로 분류기 격자 + 빼기 변형의 la (75),
    S3 대회는 s3_la_labels (la 1 · 10 · 100). 이미 있는 파일의 열 (앞선 실험이 쓴 라벨) 은 값이 그대로여야 함 → 다르면 멈춤."""
    import os

    from fab.bench.anomaly import ensure_prepared
    from fab.bench.anomaly.methods import DROP_LEVELS, LA_LEVELS

    ensure_prepared(comp)
    ad = comp.anomaly
    cond = ad.get("condition") or {}
    full = pd.read_csv(TRAIN_LABELS / f"{comp.name}.csv").set_index(ID_COL)[LABEL_COL]
    if not cond:
        t = adbench_la_labels(ad["dataset"], int(ad["seed"]), sorted({*LA_LEVELS, *DROP_LEVELS}))
    elif cond.get("exp") == "s3":
        t = s3_la_labels(comp, S3_LA)
    else:
        raise SystemExit(f"{comp.name}: la 라벨은 공식 대회 · S3 대회만")
    # train 라벨 (prepare 가 쓴 것) 과 id · 이상이 맞는지: 라벨 준 행은 모두 이상이고 la100 은 이상 전부
    if (full.loc[t[ID_COL]].to_numpy() < t["la100"].to_numpy()).any() or t["la100"].sum() != full.sum():
        raise SystemExit(f"{comp.name}: la 라벨이 train 라벨과 안 맞음")
    dst = TRAIN_LABELS / f"{comp.name}.la.csv"
    counts = {c: int(t[c].sum()) for c in t.columns if c != ID_COL}
    if dst.exists():
        old = pd.read_csv(dst)
        gone = [c for c in old.columns if c not in t.columns]
        if gone or not old[ID_COL].equals(t[ID_COL]) or any((old[c] != t[c]).any() for c in old.columns):
            raise SystemExit(f"{comp.name}: 이미 있는 la 라벨 ({dst.name}) 과 다름 (없어진 열 {gone}) → 앞선 결과가 쓴 라벨이라 덮어쓰지 않음")
        if list(old.columns) == list(t.columns):
            print(f"{comp.name}: 라벨 준 이상 {counts} (그대로)")
            return
    tmp = dst.with_name(dst.name + ".tmp")  # 다른 worktree 가 읽는 중일 수 있어 한 번에 바꿈
    t.to_csv(tmp, index=False)
    os.replace(tmp, dst)
    print(f"{comp.name}: 라벨 준 이상 {counts}")


# --- S3: 정상 하나 + 네 종류 이상 ----------------------------------------------------------------------------------


@cache
def s3_pool(dataset: str, seed: int) -> dict:
    """ADBench generate_realistic_synthetic 와 같은 순서 · 식으로 정상 하나와 네 종류 이상 (docs/anomaly.md 결정 18).
    set_seed → (1만 행 넘으면 set_seed 다시 · 1만 행 표본) → 정상 행으로 GMM (성분 1 ~ 9 중 BIC 최소, random_state=seed) →
    정상 = gm.sample(n_정상). 이상 (종류마다 원래 이상 수): global = 각 열 [min × 1.1, max × 1.1] 균등 (numpy 전역 상태, ADBench 의
    global 과 같은 자리라 먼저), local = 공분산 × 5, cluster = 평균 × 5 (둘 다 원문처럼 gm 을 고쳐 sample), dependency = 원래 정상
    행의 열마다 copulas GaussianKDE 표본 (원문 식. 정상은 원문의 vine copula 대신 GMM)."""
    from sklearn.mixture import GaussianMixture

    from fab.bench.anomaly.conditions import E1_ALPHA, E1_PERCENTAGE

    X, y = load_npz(dataset)
    np.random.seed(seed)
    random.seed(seed)
    if len(y) > MAX_ROWS:
        np.random.seed(seed)
        random.seed(seed)
        idx = np.random.choice(np.arange(len(y)), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    n_norm, n_anom = int((y == 0).sum()), int((y == 1).sum())
    Xn = X[y == 0]
    bic = [GaussianMixture(n_components=k, random_state=seed).fit(Xn).bic(Xn) for k in range(1, 10)]
    gm = GaussianMixture(n_components=int(np.argmin(bic)) + 1, random_state=seed).fit(Xn)
    normal = gm.sample(n_norm)[0]
    out = {"normal": normal}
    out["global"] = np.array([np.random.uniform(low=normal[:, i].min() * (1 + E1_PERCENTAGE),
                                                high=normal[:, i].max() * (1 + E1_PERCENTAGE), size=n_anom)
                              for i in range(normal.shape[1])]).T
    g = copy.deepcopy(gm)
    g.covariances_ = E1_ALPHA * g.covariances_
    out["local"] = g.sample(n_anom)[0]
    g = copy.deepcopy(gm)
    g.means_ = E1_ALPHA * g.means_
    out["cluster"] = g.sample(n_anom)[0]
    from copulas.univariate import GaussianKDE

    dep = np.zeros((n_anom, Xn.shape[1]))
    for i in range(Xn.shape[1]):
        kde = GaussianKDE()
        kde.fit(Xn[:, i])
        dep[:, i] = kde.sample(n_anom)
    out["dependency"] = dep
    return out


def s3_halves(dataset: str, seed: int) -> dict:
    """정상 · 종류마다 70/30 (test 는 ceil(0.3 n), sklearn train_test_split 과 같은 크기). 고정 seed 의 순열."""
    pool = s3_pool(dataset, seed)
    num = int(DATASETS[dataset].split("_", 1)[0])
    out = {}
    for k, (name, A) in enumerate(pool.items()):
        perm = np.random.default_rng([num, seed, 30 + k]).permutation(len(A))
        n_te = math.ceil(TEST_SIZE * len(A))
        out[name] = (A[perm[n_te:]], A[perm[:n_te]])
    return out


def s3_split(comp: Competition):
    """S3 대회 (train 종류 A) → (X_train, X_test, y_train, y_test, id_train, id_test, group_test). train = 정상 70% + A 이상 70%,
    test = 정상 30% + 네 종류 이상 30% 씩 (A 와 상관없이 같은 행), MinMax 는 train 으로. 행 순서는 고정 seed 로 섞음."""
    from sklearn.preprocessing import MinMaxScaler

    from fab.bench.anomaly.conditions import comp_ids

    ad = comp.anomaly
    ds, seed, A = ad["dataset"], int(ad["seed"]), ad["condition"]["train"]
    h = s3_halves(ds, seed)
    num = int(DATASETS[ds].split("_", 1)[0])
    X_tr = np.vstack([h["normal"][0], h[A][0]])
    y_tr = np.r_[np.zeros(len(h["normal"][0]), int), np.ones(len(h[A][0]), int)]
    X_te = np.vstack([h["normal"][1], *(h[t][1] for t in S3_TYPES)])
    y_te = np.r_[np.zeros(len(h["normal"][1]), int), *(np.ones(len(h[t][1]), int) for t in S3_TYPES)]
    g_te = np.array(["normal"] * len(h["normal"][1]) + [t for t in S3_TYPES for _ in range(len(h[t][1]))])
    p_tr = np.random.default_rng([num, seed, 40 + S3_TYPES.index(A)]).permutation(len(y_tr))
    p_te = np.random.default_rng([num, seed, 50]).permutation(len(y_te))
    X_tr, y_tr, X_te, y_te, g_te = X_tr[p_tr], y_tr[p_tr], X_te[p_te], y_te[p_te], g_te[p_te]
    sc = MinMaxScaler().fit(X_tr)
    ids = comp_ids(ds, seed, f"s3-{A}", len(y_tr) + len(y_te))
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te, ids[:len(y_tr)], ids[len(y_tr):], g_te


def s3_datasets() -> list[str]:
    """E1 에서 네 종류가 모두 생성된 10개 (dependency 를 뺀 landsat · PageBlocks 말고)."""
    from fab.bench.anomaly.conditions import E1_SKIP

    return [d for d in DATASETS if ("dependency", d) not in E1_SKIP]


def s3_comps(datasets: list[str] | None = None) -> list[Competition]:
    from fab.bench.anomaly.conditions import PREFIX, cond_comp

    return [cond_comp(f"{PREFIX}s3-{a}-{slug(ds)}-s{S3_SEED}", ds, S3_SEED, dict(exp="s3", train=a))
            for a in S3_TYPES for ds in (datasets or s3_datasets())]


# --- 계획 · 제출 -------------------------------------------------------------------------------------------------


def plan(exp: str, datasets: list[str] | None) -> tuple[list[Competition], list[str], str]:
    from fab.bench.anomaly.conditions import E1_TYPES, e1_comps, e2_comps, official
    from fab.bench.anomaly.methods import CLEAN, LA_LEVELS

    if exp == "s12":
        comps = official(list(SPLITS))
        if datasets:
            comps = [c for c in comps if c.anomaly["dataset"] in datasets]
        return comps, [f"{c}+la{k}" for c in CLFS for k in LA_LEVELS], "ads-s12"
    if exp == "s3":
        return s3_comps(datasets), [*(f"{c}+la100" for c in CLFS), *S3_CPU, *S3_GPU_UNSUP], "ads-s3"
    if exp == "f0":
        comps = official(list(SPLITS))
        return comps, ["fomo0d", "fomo0d" + CLEAN], "ads-f0"
    if exp == "f1":
        return e1_comps(E1_TYPES, datasets or list(DATASETS), [1]), ["fomo0d", "fomo0d" + CLEAN], "ads-f1"
    if exp == "f2":
        from fab.bench.anomaly.conditions import E2_DATASETS

        return e2_comps(datasets or E2_DATASETS), ["fomo0d"], "ads-f2"
    if exp == "l1":  # 라벨 준 이상을 context 에서 뺌: TFM 둘 + 같은 빼기를 한 기존 방법 (대조)
        comps = official([L1_SPLIT])
        if datasets:
            comps = [c for c in comps if c.anomaly["dataset"] in datasets]
        methods = [f"{L1_TFM[0]}+dropla{k}" for k in L1_LA] + [f"{L1_TFM[1]}+dropla{k}" for k in L1_KUMO_LA] + \
            [f"{m}+dropla{k}" for m in L1_CONTROL for k in L1_LA]
        return comps, methods, "ads-l1"
    if exp == "l2":  # 처음 보는 종류 + 라벨 조금: S3 대회의 분류기만 (비지도 쪽은 la 와 상관없이 이미 있는 행)
        return s3_comps(datasets), [f"{c}+la{k}" for k in L2_LA for c in CLFS], "ads-l2"
    if exp == "l3":
        comps = [c for c in s3_comps(datasets) if (c.anomaly["condition"]["train"], c.anomaly["dataset"]) in L3_COMPS]
        return comps, ["tabpfn_unsup"], "ads-l3"
    raise SystemExit(f"실험은 s12 · s3 · f0 · f1 · f2 · l1 · l2 · l3: {exp}")


def prepare(exp: str, datasets: list[str] | None):
    from fab.bench.anomaly import ensure_prepared

    comps, methods, _ = plan(exp, datasets)
    from fab.bench.anomaly.methods import uses_la

    for c in comps:
        ensure_prepared(c)
        if any(uses_la(m) for m in methods):
            write_la_labels(c)
    print(f"{exp}: 대회 {len(comps)}개 준비")


def submit(exp: str, datasets: list[str] | None, timeout: int, accelerator: str | None, yes: bool, note: str,
           only: str | None, keep: list[str] | None):
    from fab.bench.anomaly import ANSWERS, submit_comps
    from fab.bench.anomaly.methods import drop_spec, is_gpu, la_spec, uses_la

    comps, methods, label = plan(exp, datasets)
    if only:
        methods = [m for m in methods if is_gpu(m) == (only == "gpu")]
        label += f"-{only}"
    if keep:
        if bad := sorted(set(keep) - set(methods)):
            raise SystemExit(f"{exp} 계획에 없는 방법: {bad} (있는 것: {' '.join(methods)})")
        methods = [m for m in methods if m in keep]
    if not methods:
        raise SystemExit("낼 방법이 없음")
    gpu = [m for m in methods if is_gpu(m)]
    cpu = [m for m in methods if not is_gpu(m)]
    print(f"{exp}: 대회 {len(comps)}개 (예: {', '.join(c.name for c in comps[:3])} …)\n"
          f"  GPU 작업 방법 {len(gpu)}개: {' '.join(gpu) or '-'}\n  CPU 작업 방법 {len(cpu)}개: {' '.join(cpu) or '-'}")
    need = [ANSWERS / f"{c.name}.csv" for c in comps] + [TRAIN_LABELS / f"{c.name}.csv" for c in comps]
    la_cols = sorted({f"la{la_spec(m) if drop_spec(m) is None else drop_spec(m)}" for m in methods if uses_la(m)})
    if la_cols:
        need += [TRAIN_LABELS / f"{c.name}.la.csv" for c in comps]
    if exp not in ("f1", "f2"):  # F1 · F2 는 조건 실험 (2026-11-25) 때 내 Volume 에 올린 data/<대회>/ 를 그대로 씀 (upload_data 는 없는
        need += [c.data_dir / "train.csv" for c in comps]  # 폴더만 올림) → TabPFN · Kumo 와 같은 행. 정답 · train 라벨만 로컬에 있으면 됨
    if missing := [p for p in need if not p.exists()]:
        raise SystemExit(f"준비 안 된 파일 {len(missing)}개 (예: {missing[:2]}) → fab ad sup prepare {exp}")
    # la 파일에 이 계획이 읽을 열이 있는지 (옛 파일: 공식 대회에 la75, S3 대회에 la1 · la10 이 없음)
    if old := [c.name for c in comps if la_cols
               and set(la_cols) - set(pd.read_csv(TRAIN_LABELS / f"{c.name}.la.csv", nrows=0).columns)]:
        raise SystemExit(f"la 파일에 {la_cols} 열이 없는 대회 {len(old)}개 (예: {old[:2]}) → fab ad sup prepare {exp}")
    if not yes:
        print("계획만 (제출은 --yes)")
        return
    submit_comps(comps, methods, note or label, timeout, accelerator, True, label)
    print("끝난 뒤 fab remote kick (이 worktree 에서) → fab sync → fab ad sup report")


# --- 결과 · Elo --------------------------------------------------------------------------------------------------


def read_supervised() -> pd.DataFrame:
    """S3 는 한 실행에서 test 종류마다 행이 여럿 → (대회, 방법, cond) 마다 마지막 행."""
    from fab.bench.anomaly.conditions import parse_cond

    if not SUPERVISED.exists():
        return pd.DataFrame(columns=["comp", "model", "dataset", "split", "score", "aupr", "exp", "cond", "opts"])
    g = pd.read_csv(SUPERVISED)
    g["cond"] = g["cond"].fillna("")
    g = g.sort_values("timestamp", kind="stable").drop_duplicates(["comp", "model", "cond"], keep="last")
    c = g.cond.map(parse_cond)
    g["train_type"] = c.map(lambda d: d.get("train"))
    g["test_type"] = c.map(lambda d: d.get("test"))
    return g


def unsup_reference() -> pd.DataFrame:
    """공식 12 × 3 의 비지도 채점 (공식 표 + 조건 표의 kumo_unsup), (데이터셋, split, 방법) 마다."""
    from fab.bench.anomaly.conditions import read_conditions
    from fab.bench.anomaly.methods import PYOD

    off = latest_grades(pd.read_csv(GRADES))
    off = off[off.model.isin([*PYOD, "tabpfn_unsup", "fomo0d"])]
    cond = read_conditions()
    ku = cond[(cond.exp == "base") & (cond.model == "kumo_unsup")]
    return pd.concat([off, ku], ignore_index=True)[["dataset", "split", "model", "score", "aupr", "metric_error"]]


def report(rounds: int):
    from fab.bench.anomaly.methods import LA_LEVELS, SUP_ANCHOR, la_spec
    from fab.bench.anomaly.methods import base as base_of
    from fab.bench.anomaly.conditions import pool_elo

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    sup = read_supervised()
    ref = unsup_reference()
    pools = []

    # S1 · S2: 공식 대회의 분류기
    s12 = sup[(sup.exp == "base") & sup.model.str.contains(r"_clf\+la", regex=True)].copy()
    if len(s12):
        s12["clf"] = s12.model.map(base_of)
        s12["la"] = s12.model.map(la_spec)
        s12["n_labeled"] = s12.opts.str.extract(r"n_labeled=(\d+)")[0].astype(float)
        mean = s12.pivot_table(index=["dataset", "la"], columns="clf", values="score", aggfunc="mean")
        sd = s12.pivot_table(index=["dataset", "la"], columns="clf", values="score", aggfunc="std")
        nl = s12.groupby(["dataset", "la"]).n_labeled.mean().rename("n_labeled")
        print("\n== S1 · S2: split 평균 AUROC (n_labeled = 라벨 준 이상 수, split 평균)\n"
              + pd.concat([mean.round(3), nl.round(1)], axis=1).to_string())
        print("\n== S1 · S2: split 사이 표준편차\n" + sd.round(3).to_string())
        print("\n== S1 · S2: 12개 평균 AUROC (la × 분류기)\n" + mean.groupby("la").mean().round(3).to_string())
        ap = s12.pivot_table(index=["dataset", "la"], columns="clf", values="aupr", aggfunc="mean")
        print("\n== S1 · S2: 12개 평균 AUPRC (la × 분류기)\n" + ap.groupby("la").mean().round(3).to_string())
        # 비지도 최고 (split 평균 중 최고) 를 넘는 가장 작은 la
        um = ref.groupby(["dataset", "model"]).score.mean().unstack()
        best = um.max(axis=1).rename("best_unsup")
        best_m = um.idxmax(axis=1).rename("best_unsup_method")
        rows = []
        for (ds, clf), g in s12.groupby(["dataset", "clf"]):
            m = g.groupby("la").score.mean().sort_index()
            over = m[m >= best[ds]]
            rows.append(dict(dataset=ds, clf=clf, best_unsup=best[ds], method=best_m[ds],
                             first_la=over.index.min() if len(over) else np.nan,
                             n_labeled=g[g.la == over.index.min()].n_labeled.mean() if len(over) else np.nan))
        cross = pd.DataFrame(rows)
        print("\n== S2: 비지도 최고를 넘는 가장 작은 la (%) · 그때 라벨 수\n"
              + cross.pivot(index="dataset", columns="clf", values="first_la").join(best).join(best_m).to_string())
        print("\n" + cross.pivot(index="dataset", columns="clf", values="n_labeled").round(1).to_string())
        # 풀: la 마다 분류기 4개 + 비지도 참고, 기준점 lgbm_clf+la<k> = 1000
        for k in LA_LEVELS:
            g = s12[s12.la == k][["model", "dataset", "split", "metric_error"]]
            if g.empty:
                continue
            r = ref[ref.model.isin(UNSUP_REF)][["model", "dataset", "split", "metric_error"]]
            long = pd.concat([g, r], ignore_index=True).rename(columns={"model": "method"})
            pools.append(pool_elo(long, f"{SUP_ANCHOR}+la{k}", f"sup:la{k}", rounds))
        # ADBench 공개값 (같은 la 의 LGB · CatB) 과 맞춤
        if REFERENCE_SUP.exists():
            adb = pd.read_csv(REFERENCE_SUP)
            adb = adb[adb.method.isin(["LGB", "CatB"])].assign(
                clf=lambda d: d.method.map({"LGB": "lgbm_clf", "CatB": "catboost_clf"}))
            ours = s12.groupby(["dataset", "clf", "la"]).agg(ours=("score", "mean"), n=("split", "nunique")).reset_index()
            cmp = ours.merge(adb.rename(columns={"aucroc": "adbench"})[["dataset", "clf", "la", "adbench"]],
                             on=["dataset", "clf", "la"])
            cmp["diff"] = cmp.ours - cmp.adbench
            print("\n== ADBench Table D6 ~ D18 와 차 (우리 split 평균 − 공개값), |차| ≤ 0.01 인 데이터셋 수 / 12\n"
                  + cmp.assign(ok=cmp["diff"].abs() <= 0.01).groupby(["clf", "la"]).agg(
                      ok=("ok", "sum"), n=("ok", "size"), med=("diff", "median")).round(3).to_string())

    # S3: train 종류 A × test 종류 B (10개 평균). 여기 표 · 풀은 la 100% 분류기 + 비지도 그대로 (la 10% · 1% 는 아래 l2_report 의 표 ·
    # 풀로 따로 → s3:A>B 풀의 참가자가 안 바뀜)
    s3 = sup[(sup.exp == "s3") & sup.model.map(lambda m: la_spec(m) in (None, 100))]
    if len(s3):
        for m, g in s3.groupby("model"):
            w = g.pivot_table(index="train_type", columns="test_type", values="score", aggfunc="mean")
            w = w.reindex(index=S3_TYPES, columns=S3_TYPES)
            print(f"\n== S3 {m}: 행 = train 이상 종류, 열 = test 이상 종류, 데이터셋 평균 AUROC\n{w.round(3).to_string()}")
        cell = s3.pivot_table(index=["train_type", "test_type"], columns="model", values="score", aggfunc="mean")
        print("\n== S3: 칸마다 방법 (데이터셋 평균)\n" + cell.round(3).to_string())
        # 칸마다 풀 (기준점 IForest = 1000, 같은 train 을 context 로 쓴 비지도와 분류기를 한 풀에)
        for a in S3_TYPES:
            for b in S3_TYPES:
                g = s3[(s3.train_type == a) & (s3.test_type == b)]
                if len(g):
                    long = g.assign(split=g.dataset.map(lambda _: "s1"))[["model", "dataset", "split", "metric_error"]]
                    pools.append(pool_elo(long.rename(columns={"model": "method"}), "iforest", f"s3:{a}>{b}", rounds))

    pools += fomo_report(rounds)
    pools += l1_report(sup, rounds)  # 새 풀은 뒤에 (앞 풀들의 행이 파일에서 그대로)
    pools += l2_report(sup, rounds)
    pools = [p for p in pools if p is not None]
    if pools:
        res = pd.concat(pools, ignore_index=True)
        cols = ["pool", "method", "elo", "elo+", "elo-", "rank", "winrate", "imputed", "datasets"]
        res = res[cols]
        res.round({"elo": 1, "elo+": 1, "elo-": 1, "rank": 3, "winrate": 4, "imputed": 4}).to_csv(SUP_ELO, index=False)
        for p, t in res.groupby("pool", sort=False):
            print(f"\n== Elo {p}\n" + t.drop(columns="pool").round(1).to_string(index=False))
        print(f"\n{SUP_ELO}: {len(res)}행")


def fomo_report(rounds: int) -> list:
    """F0 ~ F2 (FoMo-0D) 표 · 풀. 공식 fm 풀 (fomo0d 포함) 은 fab ad elo. 여기 풀: E1 종류마다 (기존 10 + TabPFN · Kumo + FoMo-0D,
    IForest = 1000, 조건 보고의 e1:<종류> 풀에 FoMo-0D 를 더한 것)."""
    from fab.bench.anomaly.conditions import E1_TYPES, read_conditions, pool_elo
    from fab.bench.anomaly.methods import ANCHOR, CLEAN, PYOD

    out = []
    off = latest_grades(pd.read_csv(GRADES))
    cond = read_conditions()
    if (off.model == "fomo0d").any():  # F0: 공식 12 × 3
        both = pd.concat([off, cond[(cond.exp == "base") & cond.model.str.startswith("kumo_unsup")]], ignore_index=True)
        w = both.pivot_table(index="dataset", columns="model", values="score", aggfunc="mean")
        base = w[[m for m in PYOD if m in w]]
        t = pd.DataFrame({"fomo0d": w.get("fomo0d"), "fomo0d+clean": w.get("fomo0d" + CLEAN),
                          "tabpfn_unsup": w.get("tabpfn_unsup"), "tabpfn+clean": w.get("tabpfn_unsup" + CLEAN),
                          "kumo_unsup": w.get("kumo_unsup"), "kumo+clean": w.get("kumo_unsup" + CLEAN),
                          "iforest": w.get(ANCHOR), "best_base": base.max(axis=1), "best_base_method": base.idxmax(axis=1)})
        t["fomo_rank"] = pd.concat([base, w["fomo0d"]], axis=1).rank(axis=1, ascending=False)["fomo0d"]
        sd = off[off.model == "fomo0d"].groupby("dataset").score.std().rename("fomo0d_sd")
        print("\n== F0: FoMo-0D (공식 12 × 3, split 평균 AUROC. fomo_rank = 기존 10 + FoMo-0D 11개 중)\n"
              + t.join(sd).sort_values("fomo0d").round(3).to_string())
    e1 = cond[cond.exp == "e1"]
    if (e1.model == "fomo0d").any():  # F1: 이상 종류
        w = e1.pivot_table(index=["type", "dataset"], columns="model", values="score")
        cols = [c for c in ["fomo0d", "fomo0d" + CLEAN, "tabpfn_unsup", "tabpfn_unsup" + CLEAN, "kumo_unsup",
                            "kumo_unsup" + CLEAN, ANCHOR, "knn", "lof", "ocsvm"] if c in w]
        print("\n== F1: E1 종류별 평균 AUROC (데이터셋 평균)\n" + w.groupby("type")[cols].mean().round(3).to_string())
        for t_ in E1_TYPES:
            g = e1[(e1.type == t_) & ~e1.model.str.endswith(CLEAN)]
            if len(g):
                out.append(pool_elo(g.rename(columns={"model": "method"}), ANCHOR, f"f1:{t_}", rounds))
    e2 = cond[cond.exp == "e2"]
    if (e2.model == "fomo0d").any():  # F2: 오염 비율 × 배치
        e2 = e2.assign(rate=e2.rate.astype(float))
        w = e2.pivot_table(index=["dataset", "rate", "arrange"], columns="model", values="score", aggfunc="mean")
        cols = [c for c in ["fomo0d", "tabpfn_unsup", "kumo_unsup", ANCHOR, "knn", "lof"] if c in w]
        print("\n== F2: E2 뽑기 평균 AUROC\n" + w[cols].round(3).to_string())
        from fab.bench.anomaly.conditions import E2_CELLS

        if E2_CELLS.exists():
            from scipy.stats import spearmanr

            cells = pd.read_csv(E2_CELLS)[["comp", "test_masking"]]
            sc = e2.pivot_table(index=["comp", "dataset", "rate", "arrange"], columns="model", values="score").reset_index()
            r0 = sc[sc.rate == 0].set_index("dataset")
            sc = sc[sc.rate > 0].merge(cells, on="comp")
            rows = []
            for m in cols:
                drop = r0[m].reindex(sc.dataset).to_numpy() - sc[m].to_numpy()
                ok = ~np.isnan(drop)
                rows.append(dict(model=m, drop_clump=np.nanmean(drop[(sc.arrange == "clump").to_numpy()]),
                                 drop_spread=np.nanmean(drop[(sc.arrange == "spread").to_numpy()]),
                                 rho_drop_masking=spearmanr(drop[ok], sc.test_masking[ok])[0],
                                 rho_drop_rate=spearmanr(drop[ok], sc.rate[ok])[0], n=int(ok.sum())))
            print("\n== F2: r = 0 대비 떨어진 AUROC (칸 단위)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    return out


def l1_table(sup: pd.DataFrame) -> pd.DataFrame:
    """L1: (데이터셋, 방법, la) 마다 빼기 변형의 AUROC 와 같은 split 의 오염 그대로 · clean, 메운 몫 = (빼기 − 오염) / (clean − 오염)
    (E3 과 같은 정의). 오염 · clean 은 이미 있는 채점 (공식 표, kumo_unsup 은 조건 표)."""
    from fab.bench.anomaly.conditions import read_conditions
    from fab.bench.anomaly.methods import CLEAN, drop_spec
    from fab.bench.anomaly.methods import base as base_of

    d = sup[(sup.exp == "base") & (sup.split == L1_SPLIT) & sup.model.map(lambda m: drop_spec(m) is not None)].copy()
    if d.empty:
        return d
    cond = read_conditions()
    both = pd.concat([latest_grades(pd.read_csv(GRADES)), cond[cond.exp == "base"]], ignore_index=True)
    ref = both[both.split == L1_SPLIT].pivot_table(index="dataset", columns="model", values="score")
    d["method"] = d.model.map(base_of)
    d["la"] = d.model.map(drop_spec)
    d["n_dropped"] = d.opts.str.extract(r"n_dropped=(\d+)")[0].astype(float)
    d["cont"] = [ref.at[ds, m] for ds, m in zip(d.dataset, d.method)]
    d["clean"] = [ref.at[ds, m + CLEAN] for ds, m in zip(d.dataset, d.method)]
    d["fill"] = (d.score - d.cont) / (d.clean - d.cont)
    return d


def l1_report(sup: pd.DataFrame, rounds: int) -> list:
    """L1 표 · 풀. 풀 l1:la<k> (s1 의 12개, IForest = 1000): 그 la 의 빼기 변형 + 같은 la 의 분류기 4개 + 비지도 참고."""
    from fab.bench.anomaly.conditions import pool_elo
    from fab.bench.anomaly.methods import ANCHOR, la_spec
    from fab.bench.anomaly.methods import base as base_of

    d = l1_table(sup)
    if d.empty:
        return []
    out = []
    for m, g in d.groupby("method", sort=False):
        w = g.pivot(index="dataset", columns="la", values="score")
        w.insert(0, "오염", g.groupby("dataset").cont.first())
        w["clean"] = g.groupby("dataset").clean.first()
        f = g.pivot(index="dataset", columns="la", values="fill")
        print(f"\n== L1 {m}+dropla<k>: AUROC ({L1_SPLIT}. 오염 = 그대로, 열 = la %, clean = 정상만)\n{w.round(3).to_string()}"
              f"\n12개 평균\n{w.mean().round(3).to_frame().T.to_string(index=False)}")
        print(f"\n== L1 {m}+dropla<k>: 메운 몫 (빼기 − 오염) / (clean − 오염)\n{f.round(2).to_string()}"
              f"\n중앙값\n{f.median().round(2).to_frame().T.to_string(index=False)}")
    med = d.pivot_table(index="la", columns="method", values="fill", aggfunc="median")
    gain = d.assign(gain=d.score - d.cont).pivot_table(index="la", columns="method", values="gain", aggfunc="mean")
    print("\n== L1: 메운 몫의 12개 중앙값 (방법 × la)\n" + med.round(2).to_string()
          + "\n== L1: 오염 그대로 대비 12개 평균 AUROC 차 (빼기 − 오염)\n" + gain.round(3).to_string())
    for m in [m for m in L1_TFM if m in med]:
        ok = med[m][med[m] >= L1_FILL_MIN]
        print(f"H12a {m}: 메운 몫 중앙값이 {L1_FILL_MIN} 이상인 가장 작은 la = {ok.index.min() if len(ok) else '없음 (격자 안)'}")
    # 보조: 무너지는 셋에서 오염 그대로의 IForest 를 넘는 가장 작은 la (clean 으로도 못 넘으면 '없음')
    ifo = latest_grades(pd.read_csv(GRADES))
    ifo = ifo[(ifo.model == ANCHOR) & (ifo.split == L1_SPLIT)].set_index("dataset").score
    rows = []
    for (ds, m), g in d[d.dataset.isin(L1_COLLAPSED) & d.method.isin(L1_TFM)].groupby(["dataset", "method"]):
        over = g[g.score >= ifo[ds]].la
        rows.append(dict(dataset=ds, method=m, iforest=ifo[ds], clean=g.clean.iloc[0],
                         first_la=over.min() if len(over) else np.nan))
    if rows:
        print("\n== L1: 무너지는 셋에서 오염 그대로의 IForest 를 넘는 가장 작은 la (%)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    # 같은 la 의 분류기 (S2, 같은 라벨) 와: 빼기 변형이 더 높은 데이터셋 수 · 12개 평균
    clf = sup[(sup.exp == "base") & (sup.split == L1_SPLIT) & sup.model.str.contains(r"_clf\+la", regex=True)].copy()
    clf["clf"] = clf.model.map(base_of)
    clf["la"] = clf.model.map(la_spec)
    rows = []
    for (m, k), g in d[d.method.isin(L1_TFM)].groupby(["method", "la"]):
        s = g.set_index("dataset").score
        r = dict(method=m, la=k, dropla=s.mean())
        for c, h in clf[clf.la == k].groupby("clf"):
            t = h.set_index("dataset").score.reindex(s.index)
            r[c], r[f"{c}_wins"] = t.mean(), f"{int((s > t).sum())}/{int(t.notna().sum())}"
        rows.append(r)
    print("\n== L1: 같은 la 의 분류기와 (12개 평균 AUROC, <분류기>_wins = 빼기 변형이 그 분류기보다 높은 데이터셋 수)\n"
          + pd.DataFrame(rows).round(3).to_string(index=False))
    ref = unsup_reference()
    ref = ref[(ref.split == L1_SPLIT) & ref.model.isin([*UNSUP_REF, "lof"])][["model", "dataset", "split", "metric_error"]]
    for k in sorted(d.la.unique()):
        long = pd.concat([d[d.la == k][["model", "dataset", "split", "metric_error"]],
                          clf[clf.la == k][["model", "dataset", "split", "metric_error"]], ref], ignore_index=True)
        out.append(pool_elo(long.rename(columns={"model": "method"}), ANCHOR, f"l1:la{k}", rounds))
    return out


def s3_summary(s3: pd.DataFrame) -> pd.DataFrame:
    """방법마다 칸 (train 종류 × test 종류, 데이터셋 평균) 의 대각선 · 대각선 밖 평균 (최저 칸) — 문서의 S3 요약 표와 같은 계산."""
    rows = []
    for m, g in s3.groupby("model", sort=False):
        w = g.pivot_table(index="train_type", columns="test_type", values="score", aggfunc="mean").reindex(
            index=S3_TYPES, columns=S3_TYPES).to_numpy()
        n = len(S3_TYPES)
        diag = np.diag(w)
        off = w[~np.eye(n, dtype=bool)]
        off_ng = np.array([w[i, j] for i in range(n) for j in range(n) if i != j and S3_TYPES[j] != "global"])
        rows.append(dict(model=m, diag=np.nanmean(diag), diag_min=np.nanmin(diag), off=np.nanmean(off),
                         off_min=np.nanmin(off), off_no_global=np.nanmean(off_ng), units=len(g)))
    return pd.DataFrame(rows)


def l2_report(sup: pd.DataFrame, rounds: int) -> list:
    """L2 (S3 대회의 분류기를 la 10% · 1% 로) · L3 (채워진 tabpfn_unsup 의 fault 두 대회) 표와 풀 s3la<주>:A>B (칸마다, IForest = 1000:
    la 10% 분류기 + S3 의 비지도)."""
    from fab.bench.anomaly.conditions import pool_elo
    from fab.bench.anomaly.methods import la_spec
    from fab.bench.anomaly.methods import base as base_of

    s3 = sup[sup.exp == "s3"].copy()
    s3["la"] = s3.model.map(la_spec)
    out = []
    l3 = s3[(s3.model == "tabpfn_unsup") & np.array([(a, ds) in L3_COMPS for a, ds in zip(s3.train_type, s3.dataset)], bool)]
    if len(l3):
        print("\n== L3: tabpfn_unsup 의 fault 두 대회 (posinf = +inf 를 맨 위 유한 점수로 바꾼 행 수)\n"
              + l3.assign(posinf=l3.opts.str.extract(r"posinf=(\d+)")[0]).pivot_table(
                  index=["comp", "posinf"], columns="test_type", values="score").round(4).to_string())
    low = s3[s3.la.notna() & (s3.la != 100)]
    if low.empty:
        return out
    print("\n== L2: S3 요약 (칸 = 데이터셋 평균. diag = 본 종류, off = 처음 보는 종류, off_no_global = test 가 global 인 칸을 뺌)\n"
          + s3_summary(s3).sort_values("model").round(3).to_string(index=False))
    for k in sorted(low.la.unique(), reverse=True):
        cell = s3[s3.la.isna() | (s3.la == k)].pivot_table(index=["train_type", "test_type"], columns="model", values="score",
                                                           aggfunc="mean")
        print(f"\n== L2 la {int(k)}%: 칸마다 방법 (데이터셋 평균)\n" + cell.round(3).to_string())
        # 데이터셋 × 칸 단위로 비지도 TFM 이 그 분류기보다 높은 수 (S3 결과의 '152개 중 99개' 와 같은 셈)
        u = s3[s3.la.isna()].pivot_table(index=["dataset", "train_type", "test_type"], columns="model", values="score")
        c = s3[s3.la == k].assign(clf=lambda t: t.model.map(base_of)).pivot_table(
            index=["dataset", "train_type", "test_type"], columns="clf", values="score")
        both = u.join(c, how="inner")
        diag = both.index.get_level_values("train_type") == both.index.get_level_values("test_type")
        rows = [dict(unsup=um, clf=cm, n=int(both[[um, cm]].notna().all(axis=1).sum()),
                     unsup_higher=int((both[um] > both[cm]).sum()),
                     diag_unsup_higher=int((both[um] > both[cm])[diag].sum()),
                     off_unsup_higher=int((both[um] > both[cm])[~diag].sum()))
                for um in ("tabpfn_unsup", "kumo_unsup") if um in both for cm in CLFS if cm in both]
        print(f"\n== L2 la {int(k)}%: 데이터셋 × 칸 중 비지도 TFM 이 분류기보다 높은 수 (diag {int(diag.sum())} · off {int((~diag).sum())})\n"
              + pd.DataFrame(rows).to_string(index=False))
    for a in S3_TYPES:
        for b in S3_TYPES:
            g = s3[(s3.train_type == a) & (s3.test_type == b) & (s3.la.isna() | (s3.la == L2_MAIN))]
            if (g.la == L2_MAIN).any():
                long = g.assign(split="s1")[["model", "dataset", "split", "metric_error"]]
                out.append(pool_elo(long.rename(columns={"model": "method"}), "iforest", f"s3la{L2_MAIN}:{a}>{b}", rounds))
    return out


def main(argv: list[str]):
    import argparse

    ap = argparse.ArgumentParser(prog="fab ad sup", description="라벨이 있는 이상탐지 · FoMo-0D (docs/anomaly.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    exps = ["s12", "s3", "f0", "f1", "f2", "l1", "l2", "l3"]
    for cmd in ("prepare", "submit"):
        p = sub.add_parser(cmd, help="대회 · la 라벨 준비 (로컬)" if cmd == "prepare" else "계획 보기 · --yes 로 제출 + 채점 요청")
        p.add_argument("exp", choices=exps)
        p.add_argument("--datasets", nargs="+", default=None)
        if cmd == "submit":
            p.add_argument("--yes", action="store_true")
            p.add_argument("--only", choices=["cpu", "gpu"], default=None)
            p.add_argument("--methods", nargs="+", default=None, help="계획 중 이 방법들만")
            p.add_argument("--timeout", type=int, default=360, help="작업 시간 제한 (분)")
            p.add_argument("--accelerator", default=None)
            p.add_argument("--note", default="")
    r = sub.add_parser("report", help="표 · Elo → results/anomaly_supervised_elo.csv")
    r.add_argument("--rounds", type=int, default=200)
    args = ap.parse_args(argv)
    if getattr(args, "datasets", None):
        from fab.bench.anomaly import resolve

        args.datasets = [resolve(d) for d in args.datasets]
    if args.cmd == "prepare":
        prepare(args.exp, args.datasets)
    elif args.cmd == "submit":
        submit(args.exp, args.datasets, args.timeout, args.accelerator, args.yes, args.note, args.only, args.methods)
    elif args.cmd == "report":
        report(args.rounds)
