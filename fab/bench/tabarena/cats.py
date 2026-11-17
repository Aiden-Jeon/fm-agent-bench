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
BINS = [("0%", 0, 0), ("0~50%", 0, 50), ("50~90%", 50, 90), ("90% 이상", 90, 100)]  # (이름, 초과, 이하). 0% 만 = 0
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
    """데이터셋 하나의 범주 열 통계 (로컬 캐시, 새로 받지 않음)."""
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
    from fab.bench.tabarena.elo import FM_MODELS, _ours

    kumo = _ours(grades, {"kumo (ours)": FM_MODELS["kumo (ours)"]})
    pick = lambda name: ref[ref.ta_name == name][["dataset", "split", "metric_error"]]  # noqa: E731
    return {"kumo": kumo, "ta_kumo": pick("Kumo-Tabular"), "ta_icl2": pick("TabICLv2")}


def gaps(dataset: str, rows: dict[str, pd.DataFrame], ref: pd.DataFrame) -> dict:
    want = set(split_names(dataset))
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
    return [next(b for b, lo, hi in BINS if (p == 0 if hi == 0 else lo < p <= hi)) for p in pct]


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


def report(t: pd.DataFrame):
    """상관·묶음을 출력 (문서의 표가 이것)."""
    cols = [f"{k}_{s}" for k in METHODS for s in ("pct", "sd")]
    print(f"\n== 스피어만 순위 상관 (n = {len(t)}, 괄호는 p)")
    for x in ["cat_pct", "nonbin_pct", "card_max_pct", "card_max", "high_card", "rows"]:
        cells = []
        for c in cols:
            ok = t[[x, c]].dropna()
            rho, p = spearmanr(ok[x], ok[c])
            cells.append(f"{c} {rho:+.2f} ({p:.2g})")
        print(f"{x:>13}: " + " · ".join(cells))
    has = t[t.cat > 0]
    print(f"\n== 범주 열이 있는 {len(has)}개만")
    for x in ["cat_pct", "card_max_pct", "binary_pct"]:
        v = has.assign(binary_pct=100 * has.binary_cat / has.cat)
        rho, p = spearmanr(v[x], v.kumo_pct)
        rho2, p2 = spearmanr(v[x], v.kumo_sd)
        print(f"{x:>13}: kumo_pct {rho:+.2f} ({p:.2g}) · kumo_sd {rho2:+.2f} ({p2:.2g})")
    agg = dict(n=("dataset", "size"), **{c: (c, "median") for c in cols},
               fm_wins=("kumo_pct", lambda s: f"{(s < 0).sum()}/{len(s)}"))
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
        size = pd.cut(t.rows, [0, 2500, 20000, np.inf], labels=["< 2.5k", "2.5k~20k", "> 20k"])
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
