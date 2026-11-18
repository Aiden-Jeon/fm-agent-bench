"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 타깃 비의존 피처 (train 으로만 다시 fit), 126열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p03__aide-sonnet__r1__20261116-215907 (r0f0 test AUC 0.8932, Amazon 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 58106c60, 에이전트 CV AUC 0.8903). 카드: competitions/ta-amazon-employee-access/c1.md
(PR #34). 최종 제출 = LGB 0.6 · extra-trees LGB 0.2 · CatBoost 0.2 · LR 0 의 순위 섞기 (fold 평균과 전체 재학습 반반).
옮긴 것 (계산식·순서는 에이전트 그대로, best_solution.py:24-45):
  원본      9열 (범주, F0)
  빈도      원본 9 + 쌍 36 (원본 두 개씩 이은 값) 의 값 빈도 45열     LGB·LGB2·CatBoost 가 받음
  nunique   a 값마다 b 의 고유 값 수, a ≠ b 72열                   LGB·LGB2·CatBoost 가 받음
뺀 것: 쌍 36 · 삼중 28 의 교차 열 자체 (최종에서 타깃 인코딩 키·one-hot LR 입력으로만 쓰여 타깃 비의존 꼴로 안 들어감, 카드),
쌍·원본 45열의 타깃 인코딩 (OOF, 규칙 3), one-hot LR 두 개의 OOF logit (스태킹 = 타깃 사용), one-hot 인코딩 (LR 입력),
LGB·CatBoost·LR 모델과 순위 섞기·전체 재학습 섞기.
train+test fit: 에이전트는 빈도·nunique 를 train+test 전체로 셈 → 여기서는 같은 계산을 train 행으로만 fit (PR #34 의 R3 해석,
2026-11-18 사용자). test 에만 있는 값은 빈도 0 · 고유 수 0.

동일성 확인 (로컬, 2026-11-18): fit 행을 train+test 로 둔 판 (counts(al, 모든 행)) 이 PR #35 판 (이 117열 + 원본 9 로 에이전트
CatBoost 성분 CV 를 기록 0.87854 와 맞춘 판) 의 같은 열과 배열이 같은지 본 뒤, train 판과의 차이를 셈 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (126열).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module
from fab.core.features import split

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_amazon_f0")

TFM_MAX_FEATURES = None
BASE = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
        "ROLE_FAMILY", "ROLE_CODE"]


def counts(al: pd.DataFrame, fit: np.ndarray) -> pd.DataFrame:
    """에이전트 best_solution.py 의 빈도·nunique 117열 (al = train+test 의 원본 9열, fit = 셈에 쓸 행).

    에이전트는 factorize 한 번호로 셈. 번호는 문자열과 1:1 이라 문자열로 세도 값이 같음."""
    raw = al[BASE].astype(str)
    keys = {c: raw[c] for c in BASE}
    keys.update({f"{a}_{b}": raw[a] + "_" + raw[b] for a, b in combinations(BASE, 2)})
    feat = {f"{c}_cnt": s.map(s[fit].value_counts()).fillna(0) for c, s in keys.items()}
    for a in BASE:
        for b in BASE:
            if a != b:
                nu = raw[b][fit].groupby(raw[a][fit]).nunique()
                feat[f"nu_{a}_{b}"] = raw[a].map(nu).fillna(0)
    return pd.DataFrame(feat, index=al.index).astype(float)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    al = pd.concat([train[BASE], test[BASE]], keys=["train", "test"])
    fit = al.index.get_level_values(0) == "train"
    new_tr, new_te = split(counts(al, fit))
    return pd.concat([X, new_tr], axis=1), pd.concat([X_test, new_te], axis=1), cats
