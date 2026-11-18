"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 히스토그램 피처 78열을 더한 248열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p12__aide-sonnet__r1__20261117-002041 (r0f0 test AUC 0.9952, APSFailure 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 b8bde260, 에이전트 OOF AUC 0.99179). 최종 제출 = CatBoost (GPU) · XGBoost (GPU) ·
LightGBM 의 순위 평균, 세 모델 모두 같은 입력. 피처는 build 그대로:
  miss_cnt          원본 170열 중 결측 수 1열
  <g>_tot           히스토그램 무리 g (ag·ay·az·ba·cn·cs·ee, 각 <g>_000…009) 의 합 (모두 결측이면 결측) 7열
  <g>_<i>_r         무리 안 비율 = 칸 / 무리 합 (합이 0 이면 결측) 70열
빼 둔 것: 세 모델과 순위 평균. 타깃 피처 없음.
train+test 로 fit 한 것: 없음 (행마다 계산).
동일성 확인 (로컬, 2026-11-18): 이 피처로 에이전트 LightGBM 성분 (StratifiedKFold(5, random_state=42), seed f*100+s × 3, CPU) 을
다시 돌려 기록 OOF AUC 0.9912980404204428 과 비교 (docs/tabarena.md 의 "C1"). TFM 피처 상한 없음 (248열).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_aps_f0")

TFM_MAX_FEATURES = None
GROUPS = ["ag", "ay", "az", "ba", "cn", "cs", "ee"]


def hist_features(X: pd.DataFrame) -> pd.DataFrame:
    """에이전트 build 의 new (X = 원본 170열, float)."""
    new = {"miss_cnt": X.isna().sum(axis=1)}
    for g in GROUPS:
        cols = [f"{g}_{i:03d}" for i in range(10)]
        tot = X[cols].sum(axis=1, min_count=1)
        new[f"{g}_tot"] = tot
        denom = tot.replace(0, np.nan)
        for c in cols:
            new[f"{c}_r"] = X[c] / denom
    return pd.DataFrame(new, index=X.index).astype(float)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    return (pd.concat([X, hist_features(X)], axis=1), pd.concat([X_test, hist_features(X_test)], axis=1), cats)
