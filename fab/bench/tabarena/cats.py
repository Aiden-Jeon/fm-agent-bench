"""범주 열 비율과 FM − 기본 GBDT (docs/tabarena.md "범주 비율과 FM − GBDT"). 이미 있는 결과만 씀 (원격·GPU 없음).

    uv run --frozen fab ta cats              # 51개 표 → results/tabarena_cats.csv + 상관·묶음 출력

- 범주 열 = F0 이 쓰는 categorical(meta) (OpenML 범주 표시 + 문자열 열). 피처 수 = meta 의 features
- 카디널리티 = 범주 열마다 결측 (빈칸 + 글 "nan" 등, NA_TEXT) 을 뺀 고유값 수 (데이터 전체, train·test 를 가르지 않음).
  이진 = 고유값 2개인 범주 열 (결측이 있어도 값이 둘이면 이진)
- 차이 = (split 평균 metric_error − TabArena 기본 GBDT) / 기본 GBDT. 기본 GBDT 는 default_gbdt (CatBoost·LightGBM 중
  모든 split 평균이 나은 쪽, 부록과 같음). ÷σ 는 그 방법의 split 표준편차 (모든 split, ddof 1. "Agent 확장 1회째" 의 요약 표와 같음)
- kumo (ours) = kumo_cat_refit, 없으면 kumo_c8k_refit (elo.FM_MODELS 와 같은 규칙), 참고로 TabArena Kumo-Tabular · TabICLv2
- 오염 묶음 = docs/tabarena.md 6 의 표에 이름이 적힌 것 + Kaggle 데이터셋 (부록 출처 Kaggle, 대회 3개 뺌). 나머지는 미분류
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fab.bench.tabarena import (GRADES, REFERENCE, RESULTS, categorical, datasets, default_gbdt, latest_grades,
                                read_data, split_names)

CATS = RESULTS / "tabarena_cats.csv"  # git 추적, fab ta cats
HIGH_CARD = 100  # 고유값이 이보다 많은 범주 열을 고카디널리티로 셈
NA_TEXT = ["nan", "NaN", "None", "<NA>"]  # 셀 때 결측으로 보는 글
BINS = [("0%", 0, 0), ("0~50%", 0, 50), ("50~90%", 50, 90), ("90% 이상", 90, 100)]  # 0% = 0, 0~50% = (0, 50], 50~90% = (50, 90), 90% 이상 = [90, 100]
METHODS = {"kumo": "kumo (ours)", "ta_kumo": "TabArena Kumo-Tabular", "ta_icl2": "TabArena TabICLv2"}
# docs/tabarena.md 6 의 묶음 (이름이 적힌 것만). Kaggle 데이터셋 = 부록의 출처 Kaggle 중 대회 3개를 뺀 12개
CONTAM = {
    "대회": ["Amazon_employee_access", "Bioresponse", "GiveMeSomeCredit"],
    "Kaggle": ["healthcare_insurance_expenses", "Fitness_Club", "Another-Dataset-on-used-Fiat-500",
               "Is-this-a-good-customer", "Marketing_Campaign", "Bank_Customer_Churn", "heloc", "E-CommereShippingData",
               "HR_Analytics_Job_Change_of_Data_Scientists", "Food_Delivery_Time", "SDSS17",
               "customer_satisfaction_in_airline"],
    "교과서급": ["credit-g", "diabetes", "wine_quality", "bank-marketing", "houses", "diamonds",
              "blood-transfusion-service-center", "concrete_compressive_strength", "airfoil_self_noise", "splice",
              "anneal"],
    "덜 알려진": ["hazelnut-spread-contaminant-detection", "MIC", "seismic-bumps", "NATICUSdroid",
              "polish_companies_bankruptcy", "QSAR-TID-11", "jm1", "coil2000_insurance_policies"],
}
TASK = {"binary": "이진", "multiclass": "다중", "regression": "회귀"}


def cat_stats(dataset: str) -> dict:
    """데이터셋 하나의 범주 열 통계 (로컬 캐시. 캐시에 없으면 read_data → fetched 가 OpenML 에서 받음)."""
    df, meta, _ = read_data(dataset)
    cats = categorical(meta)
    # 결측이 글 "nan" 으로 든 열이 있음 (MIC 93열 · kddcup09 22열 등, F0 은 이것을 범주 값 하나로 봄) → 셀 때만 결측으로
    card = pd.Series({c: df[c][~df[c].isin(NA_TEXT)].nunique(dropna=True) for c in cats}, dtype=float)
    n, nf = len(df), len(meta["features"])
    nb = int((card > 2).sum())  # 고유값 3개 이상 (이진이 아닌) 범주 열
    out = dict(dataset=dataset, rows=n, features=nf, cat=len(cats), cat_pct=100 * len(cats) / nf,
               binary_cat=int((card == 2).sum()), nonbin_pct=100 * nb / nf, high_card=int((card > HIGH_CARD).sum()),
               card_median=card.median() if len(card) else np.nan, card_max=card.max() if len(card) else np.nan)
    out["card_max_pct"] = 100 * out["card_max"] / n if len(card) else np.nan  # 가장 큰 범주 열의 고유값 / 행 수
    return out


def method_rows(grades: pd.DataFrame, ref: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """방법마다 (dataset, split, metric_error)."""
    from fab.bench.tabarena.elo import FM_MODELS, _ours, _ref

    kumo = _ours(grades, {"kumo (ours)": FM_MODELS["kumo (ours)"]})
    r = _ref(ref)
    return {"kumo": kumo, "ta_kumo": r[r.method == "Kumo-Tabular"], "ta_icl2": r[r.method == "TabICLv2"]}


def gaps(dataset: str, rows: dict[str, pd.DataFrame], ref: pd.DataFrame) -> dict:
    want = set(split_names(dataset))
    g = ref[ref.ta_name.isin(["CatBoost", "LightGBM"]) & (ref.dataset == dataset) & ref.split.isin(want)]
    have = g.groupby("ta_name").split.nunique().reindex(["CatBoost", "LightGBM"], fill_value=0)
    if (have != len(want)).any():  # default_gbdt 는 있는 split 만 평균함 → FM 과 split 이 다르면 알림
        print(f"경고: {dataset} 의 기본 GBDT split " + " · ".join(f"{k} {v}/{len(want)}" for k, v in have.items()))
    gbdt = default_gbdt(ref, dataset, want)
    out = dict(gbdt=gbdt)
    for k, r in rows.items():
        e = r[(r.dataset == dataset) & r.split.isin(want)].metric_error
        if len(e) != len(want):
            print(f"경고: {METHODS[k]} 의 {dataset} split {len(e)}/{len(want)} → 비움")
            out[f"{k}_pct"] = out[f"{k}_sd"] = np.nan
            continue
        d = e.mean() - gbdt
        out[f"{k}_pct"], out[f"{k}_sd"] = 100 * d / gbdt, d / e.std(ddof=1)
    return out


def binned(pct: pd.Series) -> list[str]:
    def one(p):
        return "0%" if p == 0 else "0~50%" if p <= 50 else "50~90%" if p < 90 else "90% 이상"
    return [one(p) for p in pct]


def table() -> pd.DataFrame:
    ds = datasets()
    grades, ref = latest_grades(pd.read_csv(GRADES)), pd.read_csv(REFERENCE)
    rows = method_rows(grades, ref)
    contam = {d: g for g, xs in CONTAM.items() for d in xs}
    out = []
    for dataset in ds.sort_values("n_rows").index:
        out.append(dict(**cat_stats(dataset), task=TASK[ds.loc[dataset, "problem_type"]],
                        contam=contam.get(dataset, "미분류"), **gaps(dataset, rows, ref)))
    t = pd.DataFrame(out)
    t["cat_bin"], t["nonbin_bin"] = binned(t.cat_pct), binned(t.nonbin_pct)
    return t


def rho(t: pd.DataFrame, x: str, y: str) -> str:
    """스피어만 상관 "ρ (p · n)". 둘 중 하나라도 빈 행은 뺌."""
    ok = t[[x, y]].dropna()
    r, p = spearmanr(ok[x], ok[y])
    return f"{r:+.2f} ({p:.2g} · {len(ok)})"


def report(t: pd.DataFrame):
    """상관·묶음을 출력 (문서의 표가 이것)."""
    cols = [f"{k}_{s}" for k in METHODS for s in ("pct", "sd")]
    print(f"\n== 스피어만 순위 상관 ({len(t)}개 중 두 값이 다 있는 것, 괄호는 p · n)")
    for x in ["cat_pct", "nonbin_pct", "card_max_pct", "card_max", "high_card", "rows"]:
        print(f"{x:>13}: " + " · ".join(f"{c} {rho(t, x, c)}" for c in cols))
    has = t[t.cat > 0].assign(binary_pct=lambda v: 100 * v.binary_cat / v.cat)
    print(f"\n== 범주 열이 있는 {len(has)}개만")
    for x in ["cat_pct", "card_max_pct", "binary_pct"]:
        print(f"{x:>13}: kumo_pct {rho(has, x, 'kumo_pct')} · kumo_sd {rho(has, x, 'kumo_sd')}")
    print("\n== 묶어서 본 상관 (cat_pct 와 kumo_pct)")
    for task, v in t.groupby("task"):
        print(f"{'과제 ' + task:>13}: {rho(v, 'cat_pct', 'kumo_pct')}")
    for name, v in [("> 2만 행", t[t.rows > 20000]), ("≤ 2만 행", t[t.rows <= 20000])]:
        print(f"{name:>13}: {rho(v, 'cat_pct', 'kumo_pct')}")
    print(f"{'행 수':>13}: cat_pct {rho(t, 'rows', 'cat_pct')} · high_card {rho(t, 'rows', 'high_card')}")
    agg = dict(n=("dataset", "size"), **{c: (c, "median") for c in cols},
               fm_wins=("kumo_pct", lambda s: f"{(s < 0).sum()}/{s.notna().sum()}"))
    order = [b for b, _, _ in BINS]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n== 범주 비율 묶음 (중앙값)")
        print(t.groupby("cat_bin").agg(**agg).reindex(order).round(2).to_string())
        print("\n== 이진이 아닌 범주 비율 묶음 (중앙값)")
        print(t.groupby("nonbin_bin").agg(**agg).reindex(order).round(2).to_string())
        hc = np.where(t.cat == 0, "범주 없음", np.where(t.high_card > 0, f"고카디널리티 (> {HIGH_CARD}) 있음", "범주 있음, 고카디널리티 없음"))
        print("\n== 고카디널리티 묶음 (중앙값)")
        print(t.groupby(hc).agg(**agg).round(2).to_string())
        print("\n== 과제 × 범주 묶음 (kumo_pct 중앙값, 개수)")
        print(t.pivot_table(index="task", columns="cat_bin", values="kumo_pct", aggfunc=["median", "size"])
              .reindex(columns=order, level=1).round(1).to_string())
        size = pd.cut(t.rows, [0, 2499, 20000, np.inf], labels=["< 2.5k", "2.5k~20k", "> 20k"])  # 행은 정수
        print("\n== 크기 × 범주 묶음 (kumo_pct 중앙값, 개수)")
        print(t.assign(size=size).pivot_table(index="size", columns="cat_bin", values="kumo_pct",
                                              aggfunc=["median", "size"], observed=False)
              .reindex(columns=order, level=1).round(1).to_string())
        print("\n== 오염 묶음 × 범주 묶음 (개수)")
        print(pd.crosstab(t.contam, t.cat_bin).reindex(columns=order, fill_value=0).to_string())
        print("\n== 오염 묶음 (중앙값)")
        print(t.groupby("contam").agg(n=("dataset", "size"), cat_pct=("cat_pct", "median"),
                                      kumo_pct=("kumo_pct", "median"), kumo_sd=("kumo_sd", "median")).round(2)
              .to_string())


def main():
    t = table()
    t.round(4).to_csv(CATS, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 100, "display.max_columns", 30):
        print(t.drop(columns=["gbdt"]).round(2).to_string(index=False))
    report(t)
    print(f"\n{CATS}: {len(t)}행")
