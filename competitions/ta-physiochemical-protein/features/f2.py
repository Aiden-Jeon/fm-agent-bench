"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 쌍 비율 36열을 더한 45열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p04__codex-gpt__r1__20261116-230759 (r0f0 test RMSE 2.977, physiochemical 의 Agent 최고) 의 작업 디렉터리.
최종 제출 (codex.jsonl 끝의 compose_submission_nn.py 변형) = meta 모델 (CatBoost·Ridge·LightGBM, 입력 = 원본 또는 아래 피처 +
기반 모델들의 OOF 예측) · NN · CatBoost 섞기, train 과 같은 행은 train 평균을 섞고 clip(0, 21). 피처는 full_trees.py 의 features
(= grouped_base6.py 의 engineer) 그대로: 원본 9열 + i < j 쌍마다 {a}_div_{b} = a / (b + 0.0001) 36열, inf → 결측 → 0.
빼 둔 것: meta 입력의 OOF 예측 (스태킹 = 타깃), metric_feat (log1p·log 차를 RobustScaler 로 맞춘 뒤 학습한 거리 = local·kNN 모델 입력,
log 차는 비율의 단조 변환), GP·local·NN·트리 모델과 섞기, train 과 같은 행에 train 정답 섞기 (후처리), clip. 타깃 피처는 이것뿐.
train+test 로 fit 한 것: 없음 (행마다 계산).
동일성 확인 (로컬, 2026-11-18): 이 피처로 에이전트 lgb63_42 (LGBMRegressor 5400번, train 전체 fit) 를 다시 돌려 작업 디렉터리의
full_trees_preds.npz 의 test 예측과 원소별 비교 (docs/tabarena.md 의 "C1"). TFM 피처 상한 없음 (45열).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TFM_MAX_FEATURES = None


def features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """에이전트 full_trees.py 의 features 그대로."""
    out = df[cols].astype(float).copy()
    new = {f"{a}_div_{b}": df[a] / (df[b] + .0001) for i, a in enumerate(cols) for b in cols[i + 1:]}
    out = pd.concat([out, pd.DataFrame(new, index=df.index)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def build(train: pd.DataFrame, test: pd.DataFrame):
    cols = [c for c in test.columns if c != "id"]  # 에이전트의 te.columns.drop('id') (F0 의 원본 9열과 같음, 범주 없음)
    return features(train, cols).reset_index(drop=True), features(test, cols).reset_index(drop=True), []
