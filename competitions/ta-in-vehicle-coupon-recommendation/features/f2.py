"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 타깃 비의존 피처 (설문 묶음은 train 으로만 다시 fit), 75열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p02__codex-gpt__r1__20261114-122945 (파일럿 1회째, r0f0 test AUC 0.8685, in_vehicle 의 Agent 최고) 의
작업 디렉터리. 카드: competitions/ta-in-vehicle-coupon-recommendation/c1.md (PR #34, 카드의 ~f2).
최종 제출 (assemble.py, inline#317.2) = CatBoost 여러 개 (extra_long·extra_seeds·cat_more_full·template_full·make_candidate·onehot_full)
+ LR (make_candidate·linear_scenario_full) + LightGBM (lgb_full·boost_full·lgb_diverse_full) + XGBoost 의 온도 섞기.
옮긴 것 (계산식은 에이전트 그대로, 결측 범주는 'Missing'. 카드의 블록 이름):
  A 원본       24열 (F0)
  B features() train.py:14-23        person (인구통계·방문 빈도 13열을 이은 응답자 키, 행 단위), matched_frequency (범주),
                                     방문 빈도 5 + 맞는 빈도의 순서 숫자 (never 0 … gt8 4, 결측 −1), age_ordinal, person_coupon = 10열
  C 교차 cat_more_full.py:9 등       person_<상황 8>, person_demo (인구통계 8열을 이은 값), person_demo_<coupon·destination·time>,
                                     coupon_<상황 8> (범주) = 20열 (coupon_frequency 는 D 와 같아 한 번)
  D extra_features.py:4-25           income·education 순서, 방문 빈도 통계 6, 맞는 빈도 − 평균·− 최대·이하인 개수 3, scenario_full·
                                     partial·context 3, coupon_frequency (범주) = 15열
  E survey_features.py:4-31          survey_block (응답자별 시나리오 22개 템플릿으로 고른 설문 묶음), person_block, person_block_coupon
  F survey_template.py:4-27          survey_template (묶음 안 변형), person_template, person_template_coupon
  (B~F 의 글자 열은 범주. E·F 6열은 카드대로 모두 넣음: 최종 성분 코드에 실제로 들어가는 것은 survey_block·person_block·
  person_block_coupon (extra_long·extra_seeds 등) 과 survey_template (template_full·linear_scenario_full) 이고, person_template·
  person_template_coupon 은 template_full.py 가 drop 함. 2026-11-18 사용자가 카드 기준 6열로 정함)
뺀 것: 응답자별 타깃 집계 (train.py 의 add_stats, 규칙 3, 최종 제출에 없음), 응답자 × 장소 과거 정답 합 (second_stage.py 등, 최종 체인에
없음), one-hot·2차 다항 (LR 입력), 모델·섞기.
train+test fit: 에이전트는 E·F 의 템플릿 (응답자마다 22개 시나리오가 다 있는 묶음) 과 묶음 개수를 train+test 행 전체로 구함 → 같은 계산을
train 행으로만 fit (PR #34 의 R3 해석, 2026-11-18 사용자). test 행은 train 에서 정한 시나리오 → 묶음 후보와, 같은 응답자의 train 행에서
센 묶음 개수로 정하고, 못 정하면 에이전트와 같이 −1 · 'amb'. train 만 보면 22개가 다 있는 응답자가 2명뿐이라 묶음이 2개로 줄고
(train+test 로는 6개) 많은 행이 −1 임 (docs/tabarena.md 의 "C1"). B~D 는 모두 행마다 계산 (fit 없음).
동일성 확인 (로컬, 2026-11-18): fit 을 train+test 로 둔 판 (survey(x, 모든 행)) 의 E·F 가 에이전트 add_survey_template(features(train+test))
와 행마다 같은지, B~D 가 에이전트 features · add_extra_features · richify 의 출력과 같은지 (에이전트 코드를 pandas 2.3 의 object 열로 돌림),
그 열로 카드의 xgb_more (seed 11, fold 0) 를 CPU 로 다시 돌림. 그 뒤 train 판과의 차이를 셈 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (75열).
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
PERSON = ["gender", "age", "maritalStatus", "has_children", "education", "occupation", "income", "car", "Bar", "CoffeeHouse",
          "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]
DEMO = PERSON[:8]
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
CROSS = ["destination", "passenger", "time", "weather", "expiration", "toCoupon_GEQ15min", "toCoupon_GEQ25min",
         "direction_same"]


def strings(df: pd.DataFrame) -> pd.DataFrame:
    """features() 의 첫 줄: 문자열 열의 결측 → 'Missing'. 에이전트 (pandas 2.3) 의 object 열 = 숫자가 아닌 열 (pandas 3 은 str)."""
    x = df.copy()
    for col in x.select_dtypes(exclude="number").columns:
        x[col] = x[col].fillna("Missing").astype(str)
    return x


def row_features(x: pd.DataFrame) -> pd.DataFrame:
    """B·C·D: train.py 의 features, extra_features.py, cat_more_full.py 의 richify (행마다, fit 없음)."""
    new = pd.DataFrame(index=x.index)
    new["person"] = x[PERSON].astype(str).agg("|".join, axis=1)
    new["matched_frequency"] = [row[VENUES[row["coupon"]]] for _, row in x.iterrows()]
    for col in FREQ:
        new[col + "_ordinal"] = x[col].map(ORD).fillna(-1)
    new["matched_frequency_ordinal"] = new["matched_frequency"].map(ORD).fillna(-1)
    new["age_ordinal"] = x["age"].map(AGE)
    new["person_coupon"] = new["person"] + "|" + x["coupon"]
    new["income_ordinal"] = x["income"].map(INC).fillna(-1)
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
    for col in CROSS:
        new["person_" + col] = new["person"] + "|" + x[col].astype(str)
    new["person_demo"] = x[DEMO].astype(str).agg("|".join, axis=1)
    for col in ["coupon", "destination", "time"]:
        new["person_demo_" + col] = new["person_demo"] + "|" + x[col].astype(str)
    for col in CROSS:
        new["coupon_" + col] = x["coupon"] + "|" + x[col].astype(str)
    new["coupon_frequency"] = x["coupon"] + "|" + new["matched_frequency"]
    return new


def survey_block(person: pd.Series, scenario: pd.Series, fit: np.ndarray) -> pd.Series:
    """survey_features.py 의 add_survey_features 의 survey_block. 템플릿·묶음 개수는 fit 행으로만."""
    from scipy.sparse.csgraph import connected_components

    table = pd.DataFrame({"person": person, "scenario": scenario})
    g = table[fit].groupby("person").scenario
    sizes, sets = g.size(), g.agg(lambda s: tuple(sorted(s)))
    templates = sets[sizes == 22].value_counts().index.tolist()
    template_sets = [set(t) for t in templates]
    mapping: dict[str, set[int]] = {}
    if template_sets:
        adjacency = np.array([[len(a & b) >= 12 for b in template_sets] for a in template_sets], dtype=int)
        _, components = connected_components(adjacency, directed=False)
        for template, component in zip(template_sets, components):
            for s in template:
                mapping.setdefault(s, set()).add(int(component))
    table["candidates"] = table.scenario.map(lambda s: mapping.get(s, set()))
    table["unambiguous"] = table.candidates.map(lambda s: next(iter(s)) if len(s) == 1 else -1)
    known = table[fit & (table.unambiguous >= 0).to_numpy()]
    block_counts = known.groupby(["person", "unambiguous"]).size().unstack(fill_value=0)
    versions = []
    for p, candidates, initial in table[["person", "candidates", "unambiguous"]].itertuples(index=False, name=None):
        if initial >= 0:
            versions.append(initial)
            continue
        vals = block_counts.loc[p] if p in block_counts.index else pd.Series(dtype=int)
        allowed = [k for k in vals.index if (not candidates or k in candidates) and vals[k] > 0]
        versions.append(allowed[0] if len(allowed) == 1 else -1)
    return pd.Series(versions, index=table.index).astype(str)


def survey_template(person: pd.Series, block: pd.Series, scenario: pd.Series, fit: np.ndarray) -> pd.Series:
    """survey_template.py 의 add_survey_template 의 survey_template. 템플릿과 (응답자, 묶음) 마다의 시나리오 집합은 fit 행으로만."""
    from scipy.sparse.csgraph import connected_components

    table = pd.DataFrame({"person": person, "block": block, "scenario": scenario})
    g = table[fit].groupby(["person", "block"]).scenario
    sizes, sets = g.size(), g.agg(lambda s: tuple(sorted(s)))
    full = sets[sizes == 22].reset_index().drop_duplicates(["block", "scenario"])
    tsets, blocks = [set(t) for t in full["scenario"]], full["block"].tolist()
    variants: dict[int, set[str]] = {}
    vblocks: dict[int, str] = {}
    if tsets:
        adj = np.array([[blocks[i] == blocks[j] and len(a & b) >= 20 for j, b in enumerate(tsets)]
                        for i, a in enumerate(tsets)], dtype=int)
        _, components = connected_components(adj, directed=False)
        for template, blk, component in zip(tsets, blocks, components):
            variants.setdefault(component, set()).update(template)
            vblocks[component] = blk
    byblock: dict[str, list[int]] = {}
    for key, blk in vblocks.items():
        byblock.setdefault(blk, []).append(key)
    unique = {key: vals - set().union(*(variants[o] for o in byblock[vblocks[key]] if o != key)) for key, vals in variants.items()}
    seen = table[fit].groupby(["person", "block"]).scenario.agg(set)
    out = pd.Series(index=table.index, dtype=object)
    for (p, blk), grp in table.groupby(["person", "block"], sort=False):
        scenarios = seen.get((p, blk), set())
        candidates = byblock.get(blk, [])
        active = [key for key in candidates if len(unique[key] & scenarios) > 0]
        if not active and candidates:
            scores = [len(variants[key] & scenarios) for key in candidates]
            top = max(scores)
            active = [key for key, score in zip(candidates, scores) if score == top and score > 0]
        if len(active) == 1:
            out.loc[grp.index] = str(active[0])
            continue

        def variant(s, active=active):
            possible = [key for key in active if s in variants[key]]
            return str(possible[0]) if len(possible) == 1 else "amb"

        out.loc[grp.index] = grp.scenario.map(variant)
    return out.fillna("amb").astype(str)


def survey(x: pd.DataFrame, new: pd.DataFrame, fit: np.ndarray) -> pd.DataFrame:
    """E·F 6열 (x = strings() 한 원본, new = row_features 의 출력)."""
    scenario = x[SCENARIO].astype(str).agg("|".join, axis=1)
    out = pd.DataFrame(index=x.index)
    out["survey_block"] = survey_block(new["person"], scenario, fit)
    out["person_block"] = new["person"] + "|block" + out["survey_block"]
    out["person_block_coupon"] = out["person_block"] + "|" + x["coupon"]
    out["survey_template"] = survey_template(new["person"], out["survey_block"], scenario, fit)
    out["person_template"] = out["person_block"] + "|temp" + out["survey_template"]
    out["person_template_coupon"] = out["person_template"] + "|" + x["coupon"]
    return out


def agent_features(df: pd.DataFrame, fit: np.ndarray) -> pd.DataFrame:
    """B~F 의 새 열 (df = train+test 원본, fit = 설문 묶음을 fit 할 행)."""
    x = strings(df)
    new = row_features(x)
    return pd.concat([new, survey(x, new, fit)], axis=1)


def build(train: pd.DataFrame, test: pd.DataFrame, with_test: bool = False):
    """with_test=True 면 설문 묶음을 에이전트처럼 train+test 로 fit (동일성 확인용)."""
    X, X_test, cats = _f0.build(train, test)
    cols = [c for c in test.columns if c != "id"]
    df = pd.concat([train[cols], test[cols]], keys=["train", "test"])
    fit = np.ones(len(df), bool) if with_test else df.index.get_level_values(0) == "train"
    new = agent_features(df, fit)
    new_cats = [c for c in new.columns if not pd.api.types.is_numeric_dtype(new[c])]
    for c in new.columns:
        new[c] = as_category(new[c]) if c in new_cats else new[c].astype(float)
    new_tr, new_te = split(new)
    return pd.concat([X, new_tr], axis=1), pd.concat([X_test, new_te], axis=1), [*cats, *new_cats]
