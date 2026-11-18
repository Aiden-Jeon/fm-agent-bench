"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 타깃 비의존 피처, 190열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p03__aide-sonnet__r1__20261116-215907 (r0f0 test AUC 0.8932, Amazon 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 58106c60, 에이전트 CV AUC 0.8903). 최종 제출 = LGB 0.6 · extra-trees LGB 0.2 ·
CatBoost 0.2 · LR 0 의 순위 섞기 (fold 평균과 전체 재학습 반반). 계산식·순서는 에이전트 그대로:
  원본      9열 (범주, F0)
  쌍        원본 두 개씩 이은 값 36열 (범주)                          LGB 는 타깃 인코딩으로, LR 은 one-hot 으로 씀
  삼중      RESOURCE + 나머지 두 개씩 이은 값 28열 (범주)             LR2 의 one-hot 으로만 씀
  빈도      원본 9 + 쌍 36 의 값 빈도 45열 (train+test 전체)         LGB·CatBoost
  nunique   a 값마다 b 의 고유 값 수, a ≠ b 72열 (train+test 전체)   LGB·CatBoost
빼 둔 것: 쌍·원본 45열의 타깃 인코딩 (OOF, 규칙 3), one-hot LR 두 개의 OOF logit (스태킹 = 타깃 사용), one-hot 인코딩 (LR 입력),
LGB·CatBoost·LR 모델과 순위 섞기·전체 재학습 섞기. 쌍·삼중은 에이전트가 타깃 인코딩·one-hot LR 로만 모델에 넣었지만
인코딩만 빼고 교차 열 자체는 범주로 둠 (2026-11-18 사용자. tps 의 원-핫을 빼고 원래 열을 둔 것과 같은 처리).
빈도·nunique·factorize 는 train+test 전체로 계산 (타깃 비의존, 에이전트 그대로) → inductive 밖 (docs/tabarena.md 5).

동일성 확인 (로컬, 2026-11-18): 원본 9 + 빈도 45 + nunique 72 (= 에이전트 CatBoost 성분의 입력 126열) 로 에이전트 CatBoost
(StratifiedKFold(5, random_state=42), 같은 설정·seed) 를 다시 돌려 CV AUC 를 비교 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (190열).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from fab.bench.tabarena.features import as_category
from fab.core.competition import load_module
from fab.core.features import split

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_amazon_f0")

TFM_MAX_FEATURES = None
BASE = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
        "ROLE_FAMILY", "ROLE_CODE"]


def agent_features(al: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """에이전트 best_solution.py 의 피처 부분 그대로 (al = train+test 의 원본 9열). (쌍 문자열, 삼중 문자열, 빈도·nunique)."""
    raw = al[BASE].astype(str)
    pairs = pd.DataFrame({f"{a}_{b}": raw[a] + "_" + raw[b] for a, b in combinations(BASE, 2)}, index=al.index)
    others = [c for c in BASE if c != "RESOURCE"]
    tri = pd.DataFrame({f"t_{a}_{b}": raw["RESOURCE"] + "_" + raw[a] + "_" + raw[b] for a, b in combinations(others, 2)},
                       index=al.index)
    codes = pd.DataFrame({c: pd.factorize(raw[c])[0] for c in BASE}, index=al.index)  # 에이전트의 al[c] (factorize)
    codes = codes.join(pd.DataFrame({c: pd.factorize(pairs[c])[0] for c in pairs}, index=al.index))
    feat = {f"{c}_cnt": codes[c].map(codes[c].value_counts()) for c in [*BASE, *pairs]}
    for a in BASE:
        for b in BASE:
            if a != b:
                feat[f"nu_{a}_{b}"] = codes.groupby(a)[b].transform("nunique")
    return pairs, tri, pd.DataFrame(feat, index=al.index).astype(float)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    al = pd.concat([train[BASE], test[BASE]], keys=["train", "test"])
    pairs, tri, counts = agent_features(al)
    new = pd.concat([pairs.apply(as_category), tri.apply(as_category), counts], axis=1)
    new_tr, new_te = split(new)
    X = pd.concat([X, new_tr], axis=1)
    X_test = pd.concat([X_test, new_te], axis=1)
    return X, X_test, [*cats, *pairs.columns, *tri.columns]
