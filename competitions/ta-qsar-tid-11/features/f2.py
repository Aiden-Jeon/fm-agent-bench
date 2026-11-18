"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 지문 요약 3열 + PCA 128열 (둘 다 train 으로만 다시 fit) 을 더한 1,155열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p11__codex-gpt__r1__20261116-204754 (r0f0 test RMSE 0.7976, QSAR-TID-11 의 Agent 최고) 의 작업 디렉터리.
카드: competitions/ta-qsar-tid-11/c1.md (PR #34).
최종 제출 (make_submission.py) = Tanimoto·PCA 커널 SVR · LightGBM · XGBoost · Tanimoto kNN · 순서형 NN · 임계값 증강 LightGBM·XGBoost ·
쌍 모델 섞기 + 최대 Tanimoto 유사도 이동 · 최근접 Ridge 보정 · spline 보정. 옮긴 것 (계산식은 에이전트 그대로):
  bits       X.sum(1)                    켜진 비트 수                     xgbC·xgbD·임계값 증강 LGB·XGB 의 extra
  freq_dot   X @ freq                    freq = 비트마다 1 의 비율 (float32)  (xgb_cv.py:8·threshold_*_cv.py 의 같은 식)
  idf_dot    X @ (−log(clip(freq, 0.001)))
  pca_0~127  (X − 평균) 을 공분산 고유벡터 상위 128 에 투영 (큰 고유값부터)   PCA 커널 SVR 의 입력 (pca_tree_search.py:8 → pca_features.npy,
             pca_kernel_cv.py 는 앞 32열). 모델 바로 앞의 전처리도 블록으로 봄 (PR #34 의 R3 해석)
뺀 것: Tanimoto 커널·kNN·SVR (모델), 순서형 NN, 임계값 증강 (타깃 임계값으로 행 복제 = 모델 쪽), 쌍 모델, 최대 유사도 이동·최근접 Ridge·
spline 보정 (train 정답을 쓰는 후처리), 섞기, lgb_cv.py 의 X@sqrt(freq) (최종 성분 lgbB·F·G 는 extra=False 라 비트만 받음.
카드는 LGB 가 4열을 받는다고 적었지만 코드와 다름), DAE (최종 계보 밖).
train+test fit: 에이전트는 freq 와 PCA 평균·고유벡터를 train+test 로 구함 → 같은 계산을 train 행으로만 fit (PR #34 의 R3 해석,
2026-11-18 사용자). test 행은 train 의 freq·평균·고유벡터로 계산. 고유벡터 부호는 구현마다 달라서 열마다 가장 큰 성분이 양수가 되게 고정.
동일성 확인 (로컬, 2026-11-18): fit 을 train+test 로 둔 판의 PCA 를 작업 디렉터리의 pca_features.npy (5742, 128) 와 열마다 비교
(부호를 맞춘 뒤). extra 3열의 train+test 판은 PR #35 판 그대로 (에이전트 xgbC 를 다시 돌려 기록 RMSE 0.75498 과 맞춤). 그 뒤 train 판과의
차이를 셈 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (1,155열. 원본 1,024열도 상한 없이 돌림).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_qsar_tid_f0")

TFM_MAX_FEATURES = None
N_PCA = 128


def extra(X: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """에이전트 xgb_cv.py 의 extra (X = train+test 의 비트, float32, fit = freq 를 셀 행)."""
    freq = X[fit].mean(0)
    return np.stack([X.sum(1), X @ freq, X @ (-np.log(freq.clip(.001)))], axis=1)


def pca(X: np.ndarray, fit: np.ndarray, k: int = N_PCA) -> np.ndarray:
    """에이전트 pca_tree_search.py 의 xc = X − mean, eigh(xc.T @ xc), 상위 k 를 큰 것부터 (fit 행으로 평균·고유벡터)."""
    x = X.astype(np.float64)
    xc = x - x[fit].mean(0)
    _, v = np.linalg.eigh(xc[fit].T @ xc[fit])
    v = v[:, ::-1][:, :k]
    v = v * np.sign(v[np.abs(v).argmax(0), np.arange(k)])
    return xc @ v


def build(train: pd.DataFrame, test: pd.DataFrame, with_test: bool = False):
    """with_test=True 면 에이전트처럼 train+test 로 fit (동일성 확인용)."""
    X, X_test, cats = _f0.build(train, test)
    cols = [c for c in test.columns if c != "id"]
    bits = np.concatenate([train[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)])
    n = len(train)
    fit = np.arange(len(bits)) < (len(bits) if with_test else n)
    e = pd.DataFrame(extra(bits, fit), columns=["bits", "freq_dot", "idf_dot"]).astype(float)
    p = pd.DataFrame(pca(bits, fit), columns=[f"pca_{i}" for i in range(N_PCA)])
    new = pd.concat([e, p], axis=1)
    return (pd.concat([X, new.iloc[:n].reset_index(drop=True)], axis=1),
            pd.concat([X_test, new.iloc[n:].reset_index(drop=True)], axis=1), cats)
