"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 CatBoost 성분 셋의 입력 합집합, 533열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p10__codex-gpt__r1__20261116-193556 (r0f0 test AUC 0.9956, polish 의 Agent 최고) 의 작업 디렉터리.
최종 제출 (submit_current.py) = seed 10개마다 CatBoost 성분 셋 (identity 0.6·all 0.4 → ×0.95, 원본 0.05, equals 0.1) + 같은 피처의
fold·반복을 바꾼 것을 섞고, 그룹 행 (sales_to_total_assets == total_sales_to_total_assets) 은 ExtraTrees 그룹 모델·TabPFN 과 섞음,
그룹 사전 확률 1.25배, 최근접·같은 train 행의 정답으로 덮어쓰기. 옮긴 피처 (계산식은 에이전트 그대로, 결측·inf 는 결측):
  identity   identity_utils.py 의 feature_engineer: 회계 항등식 22개 × (_diff = 좌 − 우, _relative = 차 / (|좌| + |우| + 1e-10),
             _error = |relative|) 66열 + 비율 8열 (short_liability_fraction … missing_count) = 74열
  all        feature_utils.py 의 feature_engineer(x, 'all'): 접미사 (to_total_assets·to_sales·to_short_term_liabilities·
             to_total_liabilities·to_fixed_assets) 가 같은 열 두 개씩의 차 diff_<a>_<b> + 재무 18열.
             identity 와 정의가 같은 5열 (current_assets_to_assets·depreciation_to_assets·interest_to_assets·share_capital_to_assets·
             missing_count) 은 한 번만. 이름은 같고 정의가 다른 2열 (inventory_to_assets·receivables_to_assets 가 sales_to_total_assets
             기준) 은 뒤에 _s 를 붙임
  equals     equals.py: train+test 에서 두 열 값이 같은 행 수 n 이 20 ≤ n < 전체 − 20 인 쌍마다 eq_<a>__<b> (같으면 1) 와
             dif_<a>__<b> = a − b. diff_<a>_<b> 와 같은 쌍의 dif_ 는 한 번만
빼 둔 것: 모델 입력 인코딩 fillna(−1e10), CatBoost·ExtraTrees·RF 그룹 모델과 그 안의 QuantileTransformer·SelectFromModel (fold 안
fit)·TabPFN, seed·성분 섞기, 그룹 사전 확률 1.25배, 최근접 (near.npz)·같은 train 행의 정답 덮어쓰기 (train 정답을 쓰는 후처리).
그룹 표시는 따로 넣지 않음 (모델 나누기에만 씀. 그 쌍이 equals 에 들면 eq_ 열로 들어감). 타깃 피처 없음.
train+test 로 fit 한 것: equals 의 쌍 고르기 (값이 같은 행 수를 train+test 로 셈). 나머지는 행마다 계산.
동일성 확인 (로컬, 2026-11-18): 열 수가 에이전트 기록과 같은지 (identity 138 = 원본 64 + 74, all 300, equals 362 · 쌍 112).
identity 블록으로 에이전트 CatBoost (seed 123, depth 5, 1600번) 를 다시 돌려 predictions/identity_cat5_1600_123.npz 의 oof 와
비교 (docs/tabarena.md 의 "C1"). TFM 피처 상한 없음 (TabArena 의 FM 은 1,777열까지 돎).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TFM_MAX_FEATURES = None
GROUPS = ["to_total_assets", "to_sales", "to_short_term_liabilities", "to_total_liabilities", "to_fixed_assets"]
SAME_AS_IDENTITY = ["current_assets_to_assets", "depreciation_to_assets", "interest_to_assets", "share_capital_to_assets",
                    "missing_count"]


def identity_features(x: pd.DataFrame) -> dict[str, pd.Series]:
    """identity_utils.py 의 feature_engineer 가 더하는 74열 그대로."""
    ex = {}
    c = x.__getitem__

    def identity(name, left, right):
        ex[name + "_diff"] = left - right
        ex[name + "_relative"] = (left - right) / (abs(left) + abs(right) + 1e-10)
        ex[name + "_error"] = abs(ex[name + "_relative"])

    cur = c("current_assets_to_short_term_liabilities") * c("short_term_liabilities_to_total_assets")
    cur2 = c("working_capital_to_total_assets") + c("short_term_liabilities_to_total_assets")
    sales = c("total_sales_to_total_assets")
    debt = c("total_liabilities_to_total_assets")
    short = c("short_term_liabilities_to_total_assets")
    expense = c("operating_expenses_to_short_term_liabilities") * short
    profit = c("operating_profit_to_total_assets")
    identity("expense_liabilities_consistency", expense, c("operating_expenses_to_total_liabilities") * debt)
    identity("profit_liabilities_consistency", profit, c("operating_expenses_to_total_liabilities") * debt)
    identity("op_sales_consistency", c("operating_profit_to_sales") * sales, profit)
    identity("net_sales_consistency", c("net_profit_to_sales") * sales, c("net_profit_to_total_assets"))
    identity("gross_sales_consistency", c("gross_profit_to_sales") * sales, c("gross_profit_to_total_assets"))
    identity("salesprofit_sales_consistency", c("sales_profit_to_sales") * sales, c("sales_profit_to_total_assets"))
    identity("revenue_consistency", c("sales_to_short_term_liabilities") * short, sales)
    identity("margin_sales_consistency", c("gross_margin") * sales, c("sales_profit_to_total_assets"))
    identity("cost_sales_consistency", c("total_costs_to_total_sales") * sales, expense)
    identity("margin_cost_consistency", 1 - c("total_costs_to_total_sales"), c("gross_margin"))
    identity("growth_cost_consistency", c("sales_to_total_assets") * c("total_costs_to_total_sales"), 1)
    identity("currentassets_consistency", cur, cur2)
    identity("currentassets2_consistency", c("current_assets_to_total_liabilities") * debt, cur2)
    identity("gross_shortdebt_consistency", c("gross_profit_to_short_term_liabilities") * short,
             c("gross_profit_to_total_assets"))
    identity("sales_profit_op_profit_consistency", c("sales_profit_to_total_assets"), profit)
    identity("gross_ebit_consistency", c("gross_profit_to_total_assets"), c("ebit_to_total_assets"))
    identity("ebit_extended_consistency", c("ebit_to_total_assets"), c("extended_profit_to_total_assets"))
    identity("expense_profit_revenue_consistency", expense + profit, sales)
    identity("receivables_consistency", c("receivables_turnover_ratio") * c("receivables_days_ratio"), 365)
    identity("liabilities_days_consistency", c("short_term_liabilities_days_ratio") * c("sales_to_short_term_liabilities"),
             365)
    identity("inventory_consistency", c("inventory_turnover_ratio") * c("inventory_days_ratio"), 365)
    identity("liquid_assets_consistency", c("current_assets_minus_inventory_to_short_term_liabilities") * short,
             c("working_capital_to_total_assets") + short - c("inventory_days_ratio") * sales / 365)
    ex["short_liability_fraction"] = short / (debt.abs() + 1e-10)
    ex["current_assets_to_assets"] = cur2
    ex["depreciation_to_assets"] = c("ebitda_to_total_assets") - c("ebit_to_total_assets")
    ex["interest_to_assets"] = c("gross_profit_plus_interest_to_total_assets") - c("gross_profit_to_total_assets")
    ex["share_capital_to_assets"] = c("equity_to_total_assets") - c("equity_minus_share_capital_to_total_assets")
    ex["inventory_to_assets"] = c("inventory_days_ratio") * sales / 365
    ex["receivables_to_assets"] = c("receivables_days_ratio") * sales / 365
    ex["missing_count"] = x.isna().sum(axis=1)
    return ex


def difference_pairs(cols: list[str]) -> list[tuple[str, str]]:
    return [p for g in GROUPS for p in itertools.combinations([c for c in cols if c.endswith(g)], 2)]


def all_features(x: pd.DataFrame) -> dict[str, pd.Series]:
    """feature_utils.py 의 feature_engineer(x, 'all') 가 더하는 열 그대로 (이름은 에이전트 것)."""
    extra = {f"diff_{a}_{b}": x[a] - x[b] for a, b in difference_pairs(list(x.columns))}
    c = x.__getitem__
    extra["accounting_balance"] = c("equity_to_total_assets") + c("total_liabilities_to_total_assets") - 1
    extra["long_debt_to_assets"] = c("total_liabilities_to_total_assets") - c("short_term_liabilities_to_total_assets")
    extra["short_fraction_liabilities"] = c("short_term_liabilities_to_total_assets") / (
        c("total_liabilities_to_total_assets").abs() + 0.00001)
    extra["current_assets_to_assets"] = c("working_capital_to_total_assets") + c("short_term_liabilities_to_total_assets")
    extra["quick_assets_to_assets"] = (c("current_assets_minus_inventory_to_short_term_liabilities")
                                       * c("short_term_liabilities_to_total_assets"))
    extra["liquid_assets_to_assets"] = c("liquid_assets_to_short_term_liabilities") * c("short_term_liabilities_to_total_assets")
    extra["inventory_to_current_liabilities"] = (c("current_assets_to_short_term_liabilities")
                                                 - c("current_assets_minus_inventory_to_short_term_liabilities"))
    extra["share_capital_to_assets"] = c("equity_to_total_assets") - c("equity_minus_share_capital_to_total_assets")
    extra["gross_net_difference"] = c("gross_profit_to_total_assets") - c("net_profit_to_total_assets")
    extra["depreciation_to_assets"] = c("ebitda_to_total_assets") - c("ebit_to_total_assets")
    extra["interest_to_assets"] = c("gross_profit_plus_interest_to_total_assets") - c("gross_profit_to_total_assets")
    extra["inventory_to_assets"] = c("inventory_days_ratio") * c("sales_to_total_assets") / 365
    extra["receivables_to_assets"] = c("receivables_days_ratio") * c("sales_to_total_assets") / 365
    extra["net_liabilities_to_assets"] = c("net_liabilities_to_sales") * c("sales_to_total_assets")
    extra["profit_after_dep_to_assets"] = (c("net_profit_plus_depreciation_to_total_liabilities")
                                           * c("total_liabilities_to_total_assets"))
    extra["gross_after_dep_to_assets"] = (c("gross_profit_plus_depreciation_to_total_liabilities")
                                          * c("total_liabilities_to_total_assets"))
    extra["liability_to_equity"] = c("total_liabilities_to_total_assets") / (c("equity_to_total_assets").abs() + 0.00001)
    extra["missing_count"] = x.isna().sum(axis=1)
    extra["zero_count"] = (x == 0).sum(axis=1)
    return extra


def equal_pairs(total: pd.DataFrame) -> list[tuple[str, str]]:
    """equals.py 의 쌍 고르기 (total = train+test 원본)."""
    out = []
    for a, b in itertools.combinations(total.columns, 2):
        n = (total[a].notna() & total[b].notna() & total[a].eq(total[b])).sum()
        if 20 <= n < len(total) - 20:
            out.append((a, b))
    return out


def features(x: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    ident = identity_features(x)
    allf = all_features(x)
    for k in SAME_AS_IDENTITY:
        del allf[k]
    allf["inventory_to_assets_s"] = allf.pop("inventory_to_assets")
    allf["receivables_to_assets_s"] = allf.pop("receivables_to_assets")
    eq = {}
    for a, b in pairs:
        eq[f"eq_{a}__{b}"] = (x[a].notna() & x[b].notna() & x[a].eq(x[b])).astype(int)
        if f"diff_{a}_{b}" not in allf:
            eq[f"dif_{a}__{b}"] = x[a] - x[b]
    out = pd.concat([x, pd.DataFrame({**ident, **allf, **eq}, index=x.index)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).astype(float)


def build(train: pd.DataFrame, test: pd.DataFrame):
    cols = [c for c in test.columns if c != "id"]  # 원본 64열 (모두 수치, F0 와 같음)
    x, xt = train[cols].astype(float), test[cols].astype(float)
    pairs = equal_pairs(pd.concat([x, xt], ignore_index=True))
    return features(x, pairs).reset_index(drop=True), features(xt, pairs).reset_index(drop=True), []
