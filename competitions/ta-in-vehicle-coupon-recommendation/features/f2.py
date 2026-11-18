"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 피처 중 응답자 id 계열을 뺀 것, 55열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p02__codex-gpt__r1__20261114-122945 (파일럿 1회째, r0f0 test AUC 0.8685, in_vehicle 의 Agent 최고) 의
작업 디렉터리. 최종 제출 (assemble.py) = CatBoost 여러 개 (extra_long·cat_more_full·template_full·candidate_cat·onehot_full) +
LR (candidate_lin·linear_scenario_full) + LightGBM (lgb_full·boost_full·lgb_diverse_full) + XGBoost (boost_full) 의 온도 섞기.
피처는 train.py 의 features · extra_features.py · 성분 스크립트의 교차 열. 계산식은 에이전트 그대로 (결측 범주는 'Missing'):
  원본       24열 (F0)
  맞는 빈도  쿠폰 종류에 맞는 방문 빈도 열의 값 (matched_frequency, 범주) 1열
  순서       방문 빈도 5 + 맞는 빈도의 순서 숫자 (never 0 … gt8 4, 결측 −1), 나이·소득·학력 순서 숫자 = 9열
  빈도 통계  방문 빈도 5개의 평균·표준편차·최대·최소·합·개수, 맞는 빈도 − 평균 · − 최대 · 이하인 개수 = 9열
  상황 교차  상황 10열 · (쿠폰·목적지·동승자·시각) · (목적지·동승자·시각·날씨·기온) 을 이은 값 3열 (범주)
  쿠폰 교차  쿠폰 × (목적지·동승자·시각·날씨·유효기간·15분·25분·같은 방향) 8 + 쿠폰 × 맞는 빈도 1 = 9열 (범주)
빼 둔 것 (2026-11-18 사용자): 응답자 id 계열 전부 — person (인구통계·방문 빈도 13열을 이은 값 = 응답자 id), person_coupon,
person_<상황> 8, person_demo (인구통계 8열을 이은 값) 와 그 교차, survey_block·survey_template·person_block·person_template
(train+test 를 합쳐 응답자별 설문 묶음을 다시 찾은 것). 에이전트가 응답자 id 를 train+test 로 다시 만든 풀이라 TabArena 의
"inductive" 밖이고 (docs/tabarena.md 의 "파일럿 결과"), 점수가 피처가 아닌 데서 나온 몫으로 봄 (nomad 의 정답 섞기를 뺀 것과 같은 처리).
응답자별 타깃 집계 (train.py 의 add_stats, 규칙 3), one-hot·2차 다항 (LR 입력), 모델·섞기도 뺌.
train+test 로 fit 한 것: 없음 (남은 열은 모두 행마다 계산).

동일성 확인: 피처 배열이 작업 디렉터리에 없고, 남은 모델 성분은 모두 person 계열을 써서 그대로 다시 돌릴 수 없음 →
열 이름·식을 에이전트 코드와 맞춤 (docs/tabarena.md 의 "C1"). TFM 피처 상한 없음 (55열).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.bench.tabarena.features import as_category
from fab.core.competition import load_module
from fab.core.features import split

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_in_vehicle_f0")

TFM_MAX_FEATURES = None
VENUES = {"Bar": "Bar", "Coffee House": "CoffeeHouse", "Carry out & Take away": "CarryAway",
          "Restaurant(<20)": "RestaurantLessThan20", "Restaurant(20-50)": "Restaurant20To50"}
FREQ = ["Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]
ORD = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
AGE = {"below21": 18, "21": 21, "26": 26, "31": 31, "36": 36, "41": 41, "46": 46, "50plus": 53}
INC = {"Less than $12500": 0, "$12500 - $24999": 1, "$25000 - $37499": 2, "$37500 - $49999": 3, "$50000 - $62499": 4,
       "$62500 - $74999": 5, "$75000 - $87499": 6, "$87500 - $99999": 7, "$100000 or More": 8}
EDU = {"Some High School": 0, "High School Graduate": 1, "Some college - no degree": 2, "Associates degree": 3,
       "Bachelors degree": 4, "Graduate degree (Masters or Doctorate)": 5}
SCENARIO = ["destination", "passenger", "weather", "temperature", "time", "coupon", "expiration", "toCoupon_GEQ15min",
            "toCoupon_GEQ25min", "direction_same"]
COUPON_X = ["destination", "passenger", "time", "weather", "expiration", "toCoupon_GEQ15min", "toCoupon_GEQ25min",
            "direction_same"]


def agent_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """train.py 의 features + extra_features.py + 성분의 쿠폰 교차에서 person 계열을 뺀 새 열 (df = train+test 원본)."""
    x = df.copy()
    # features(): 문자열 열의 결측 → 'Missing'. 에이전트 (pandas 2.3) 의 object 열 = 숫자가 아닌 열 (pandas 3 은 문자열이 str)
    for col in x.select_dtypes(exclude="number").columns:
        x[col] = x[col].fillna("Missing").astype(str)
    new = pd.DataFrame(index=x.index)
    new["matched_frequency"] = [row[VENUES[row["coupon"]]] for _, row in x.iterrows()]
    for col in FREQ:
        new[col + "_ordinal"] = x[col].map(ORD).fillna(-1)
    new["matched_frequency_ordinal"] = new["matched_frequency"].map(ORD).fillna(-1)
    new["age_ordinal"] = x["age"].map(AGE)
    new["income_ordinal"] = x["income"].map(INC).fillna(-1)  # extra_features.py
    new["education_ordinal"] = x["education"].map(EDU).fillna(-1)
    freq = new[[c + "_ordinal" for c in FREQ]].replace(-1, np.nan)
    new["frequency_mean"] = freq.mean(axis=1).fillna(-1)
    new["frequency_std"] = freq.std(axis=1).fillna(-1)
    new["frequency_max"] = freq.max(axis=1).fillna(-1)
    new["frequency_min"] = freq.min(axis=1).fillna(-1)
    new["frequency_sum"] = freq.sum(axis=1)
    new["frequency_count"] = freq.notna().sum(axis=1)
    match = new["matched_frequency_ordinal"].replace(-1, np.nan)
    new["frequency_matched_relative"] = match.sub(freq.mean(axis=1)).fillna(-10)
    new["frequency_matched_maxdiff"] = match.sub(freq.max(axis=1)).fillna(-10)
    new["frequency_matched_relrank"] = freq.le(match, axis=0).sum(axis=1).where(match.notna(), -1)
    new["scenario_full"] = x[SCENARIO].astype(str).agg("|".join, axis=1)
    new["scenario_partial"] = x[["coupon", "destination", "passenger", "time"]].astype(str).agg("|".join, axis=1)
    new["scenario_context"] = x[["destination", "passenger", "time", "weather", "temperature"]].astype(str).agg(
        "|".join, axis=1)
    for col in COUPON_X:  # make_candidate.py·extra_long.py·lgb_full.py 등
        new["coupon_" + col] = x["coupon"] + "|" + x[col].astype(str)
    new["coupon_frequency"] = x["coupon"] + "|" + new["matched_frequency"]
    cats = ["matched_frequency", "scenario_full", "scenario_partial", "scenario_context",
            *["coupon_" + c for c in COUPON_X], "coupon_frequency"]
    return new, cats


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    cols = [c for c in test.columns if c != "id"]
    new, new_cats = agent_features(pd.concat([train[cols], test[cols]], keys=["train", "test"]))
    for c in new.columns:
        new[c] = as_category(new[c]) if c in new_cats else new[c].astype(float)
    new_tr, new_te = split(new)
    return pd.concat([X, new_tr], axis=1), pd.concat([X_test, new_te], axis=1), [*cats, *new_cats]
