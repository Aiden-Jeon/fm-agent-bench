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


def write_la_labels(comp: Competition):
    """<대회>.la.csv. 공식 대회는 ADBench 와 같은 고르기 (adbench_la_labels), S3 대회는 la100 (train 의 A 이상 모두) 만."""
    from fab.bench.anomaly import ensure_prepared
    from fab.bench.anomaly.methods import LA_LEVELS

    ensure_prepared(comp)
    ad = comp.anomaly
    cond = ad.get("condition") or {}
    if not cond:
        t = adbench_la_labels(ad["dataset"], int(ad["seed"]), LA_LEVELS)
        # 공식 대회의 train 라벨 (prepare 가 쓴 것) 과 id · 이상이 맞는지
        full = pd.read_csv(TRAIN_LABELS / f"{comp.name}.csv").set_index(ID_COL)[LABEL_COL]
        if (full.loc[t[ID_COL]].to_numpy() < t["la100"].to_numpy()).any() or t["la100"].sum() != full.sum():
            raise SystemExit(f"{comp.name}: la 라벨이 train 라벨과 안 맞음")
    elif cond.get("exp") == "s3":
        full = pd.read_csv(TRAIN_LABELS / f"{comp.name}.csv")
        t = pd.DataFrame({ID_COL: full[ID_COL], "la100": full[LABEL_COL].astype(int)})
    else:
        raise SystemExit(f"{comp.name}: la 라벨은 공식 대회 · S3 대회만")
    t.to_csv(TRAIN_LABELS / f"{comp.name}.la.csv", index=False)
    counts = {c: int(t[c].sum()) for c in t.columns if c != ID_COL}
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
    raise SystemExit(f"실험은 s12 · s3 · f0 · f1 · f2: {exp}")


def prepare(exp: str, datasets: list[str] | None):
    from fab.bench.anomaly import ensure_prepared

    comps, methods, _ = plan(exp, datasets)
    from fab.bench.anomaly.methods import is_supervised

    for c in comps:
        ensure_prepared(c)
        if any(is_supervised(m) for m in methods):
            write_la_labels(c)
    print(f"{exp}: 대회 {len(comps)}개 준비")


def submit(exp: str, datasets: list[str] | None, timeout: int, accelerator: str | None, yes: bool, note: str,
           only: str | None, keep: list[str] | None):
    from fab.bench.anomaly import ANSWERS, submit_comps
    from fab.bench.anomaly.methods import is_gpu, is_supervised

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
    if any(is_supervised(m) for m in methods):
        need += [TRAIN_LABELS / f"{c.name}.la.csv" for c in comps]
    if exp not in ("f1", "f2"):  # F1 · F2 는 조건 실험 (2026-11-25) 때 내 Volume 에 올린 data/<대회>/ 를 그대로 씀 (upload_data 는 없는
        need += [c.data_dir / "train.csv" for c in comps]  # 폴더만 올림) → TabPFN · Kumo 와 같은 행. 정답 · train 라벨만 로컬에 있으면 됨
    if missing := [p for p in need if not p.exists()]:
        raise SystemExit(f"준비 안 된 파일 {len(missing)}개 (예: {missing[:2]}) → fab ad sup prepare {exp}")
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

    # S3: train 종류 A × test 종류 B (10개 평균)
    s3 = sup[sup.exp == "s3"]
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


def main(argv: list[str]):
    import argparse

    ap = argparse.ArgumentParser(prog="fab ad sup", description="라벨이 있는 이상탐지 · FoMo-0D (docs/anomaly.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    exps = ["s12", "s3", "f0", "f1", "f2"]
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
