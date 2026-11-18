"""C1 의 F2: TabArena Agent AIDE × GPT 풀이의 수치 열 결측 지시 17열을 더한 128열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p06__aide-gpt__r1__20261116-145653 (r0f0 test logloss 0.4445, MIC 의 Agent 최고) 의 최종 풀이
(workspace/workspaces/exp/best_solution/solution.py = journal best 노드 dedb5530, step 25). 카드: competitions/ta-mic/c1.md
(PR #34, 카드의 ~f2m. 변형 이름은 이 저장소 관례대로 ~f2).
최종 제출 = 사망 확률 (CatBoost·XGBoost 트리 섞기 + 선형 사망 LR 0.30, 로지스틱 보정) × 원인 분포 (CatBoost + XGBoost). 옮긴 것:
  <열>_missing  수치 17열 (AGE · 혈압 4 · 혈액 7 · ROE · NA_R 3 · NOT_NA 2, solution.py:371-391 의 numeric_features) 의 결측 지시
               (solution.py:409 의 MissingIndicator(features="all"): 모든 수치 열에 지시 열, fit 없음. 카드는 "결측이 있는 열만"
               으로 적었지만 코드는 "all". r0f0 train 에서 17열 모두 결측이 있어 열 수는 같음). 선형 성분의 입력 → 최종 사망 확률에
               0.30 으로 들어감. 결측은 에이전트처럼 to_numeric (coerce) 한 뒤 inf 도 결측으로 셈
뺀 것: 수치 17 의 중앙값 대치 + 표준화, 범주 94 의 −999 대치 + OneHot (선형 모델 입력 인코딩, 카드의 ~f2lr 은 1차 후보 아님),
CatBoost 입력의 결측 → −999 (모델 입력), 사망·원인 두 단계 구조 (dead 라벨 = 타깃), 보정·섞기 무게 (OOF 로 고름), ExtraTrees 원인 모델
(최종에 안 들어감). F0 의 범주 94열은 그대로 범주 (에이전트 CatBoost 는 범주 표시 없이 수치로 넘겼지만 그것은 모델 설정).
train+test fit: 없음 (행마다).
동일성 확인 (로컬, 2026-11-18): 지시 열이 에이전트 MissingIndicator(features="all") 의 출력과 같은지, solution.py 를 r0f0 데이터로 돌려
카드의 Tree CV 0.413678 → 선형 더한 0.409726 이 나오는지 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (128열).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_mic_f0")

TFM_MAX_FEATURES = None
NUMERIC = ["AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT", "K_BLOOD", "NA_BLOOD", "ALT_BLOOD", "AST_BLOOD",
           "KFK_BLOOD", "L_BLOOD", "ROE", "NA_R_1_n", "NA_R_2_n", "NA_R_3_n", "NOT_NA_2_n", "NOT_NA_3_n"]


def missing(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERIC].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.isna().astype(float).add_suffix("_missing").reset_index(drop=True)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    return pd.concat([X, missing(train)], axis=1), pd.concat([X_test, missing(test)], axis=1), cats
