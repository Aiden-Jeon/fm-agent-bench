"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 지문 요약 3열을 더한 1,027열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p11__codex-gpt__r1__20261116-204754 (r0f0 test RMSE 0.7976, QSAR-TID-11 의 Agent 최고) 의 작업 디렉터리.
최종 제출 (make_submission.py) = Tanimoto·PCA 커널 SVR · LightGBM · XGBoost · Tanimoto kNN · 순서형 NN · 임계값 증강 LightGBM·XGBoost ·
쌍 모델 섞기 + 최대 Tanimoto 유사도 이동 · 최근접 Ridge 보정 · spline 보정. 표형 성분 중 xgbC·xgbD·임계값 증강 모델이 쓰는 extra 3열
(xgb_cv.py·threshold_augmented_cv.py 의 같은 식) 을 옮김:
  bits       X.sum(1)                    켜진 비트 수
  freq_dot   X @ freq                    freq = 비트마다 1 의 비율 (train+test 전체, float32)
  idf_dot    X @ (−log(clip(freq, 0.001)))
빼 둔 것: Tanimoto 커널·kNN·SVR, PCA 커널 (pca_features.npy, 커널 SVR 입력), 순서형 NN, 임계값 증강 (타깃 임계값으로 행 복제 = 모델 쪽),
최대 유사도 이동·최근접 Ridge·spline 보정 (train 정답을 쓰는 후처리), 섞기, lgb_cv.py 의 X@sqrt(freq) (최종 성분 lgbB·F·G 는 extra 없음).
train+test 로 fit 한 것: freq (비트 빈도).
동일성 확인 (로컬, 2026-11-18): 이 피처로 에이전트 xgbC (KFold(5, random_state=2026), early stopping) 를 다시 돌려 작업 디렉터리의
xgb_cv.npz 의 oof_xgbC 와 원소별 비교, 기록 RMSE 0.7549845281594365 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (1,027열. 원본 1,024열도 상한 없이 돌림).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_qsar_tid_f0")

TFM_MAX_FEATURES = None


def extra(X: np.ndarray) -> np.ndarray:
    """에이전트 xgb_cv.py 의 extra 그대로 (X = train+test 의 비트, float32)."""
    freq = X.mean(0)
    return np.stack([X.sum(1), X @ freq, X @ (-np.log(freq.clip(.001)))], axis=1)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    cols = [c for c in test.columns if c != "id"]
    bits = np.concatenate([train[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)])
    e = pd.DataFrame(extra(bits), columns=["bits", "freq_dot", "idf_dot"]).astype(float)
    n = len(train)
    return (pd.concat([X, e.iloc[:n].reset_index(drop=True)], axis=1),
            pd.concat([X_test, e.iloc[n:].reset_index(drop=True)], axis=1), cats)
