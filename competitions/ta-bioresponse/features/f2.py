"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 PCA 30열 + 행 요약 3열 (train 으로만 fit) 을 더한 1,809열 (docs/tabarena.md 의 "행 추가: Bioresponse").

출처: 실행 tabarena-p18__aide-sonnet__r1__20261121-072650 (r0f0 test 1 − AUC 0.10900) 의 최종 풀이 = 노드 0e7a8d44.
이 실행은 작업 시간 상한에 걸려 진행 사본으로 회수됨 (run_crashed): 작업 디렉터리 사본 (00:05Z) 에는 이 노드가 없어서 코드는
aide.verbose.log 의 그 노드 응답 (00:01:28Z, AIDE 가 그대로 실행한 코드) 에서 읽음. 카드: competitions/ta-bioresponse/c1.md.
이 행의 Agent 1·2등 (Codex × GPT 0.10794 · AIDE × GPT 0.10797) 은 최종 제출에 원본 열 말고 피처 블록이 없어 R3 (b) 로 내려옴 (카드).
최종 제출 = LGB · LGB(aug) · RBF SVM 둘 (원본 · sqrt) · ExtraTrees · XGB · CatBoost 의 순위 가중 평균. 옮긴 것 (계산식은 에이전트 그대로, aug):
  pca_0~29   PCA(30, random_state=0) 을 train 행으로 fit 해 투영 (에이전트도 fold 의 train 으로 fit, 최종은 train 전체)
  bin_sum    0/1 열 (train 에서 값이 0·1 뿐인 열) 의 행 합
  cont_mean  나머지 열의 행 평균
  cont_std   나머지 열의 행 표준편차 (numpy 기본 ddof 0)
LGB(aug) 성분 하나의 입력 (원본 + 이 33열). 뺀 것: sqrt 변환 (같은 열의 단조 변환, RBF SVM 하나의 입력), 모델 · seed 평균 · 순위 섞기 무게
(OOF 정답으로 고름). 타깃 의존 블록 · train+test fit 블록은 없음.
동일성 확인 (로컬): 부모 노드 66e16ce3 (같은 aug, seed 3개) 의 lgb_aug 를 같은 fold · 설정으로 우리 코드로 다시 돌려 기록
CV AUC 0.8548138808094561 과 견줌 (docs/tabarena.md). TFM 피처 상한 없음 (1,809열. 원본 1,776열도 상한 없이 돌림).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_bioresponse_f0")

TFM_MAX_FEATURES = None
N_PCA = 30


def binary_cols(A: np.ndarray) -> np.ndarray:
    """train 에서 값이 0·1 뿐인 열 (에이전트의 isbin, train 전체로 셈)."""
    return ((A == 0) | (A == 1)).all(axis=0)  # 에이전트의 set(np.unique(col)) <= {0, 1} 와 같음


def blocks(A: np.ndarray, B: np.ndarray, isbin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """에이전트 aug 의 덧붙인 33열. A = fit 하는 행 (train), B = 변환만 하는 행. float32 (에이전트와 같음).
    isbin 을 주면 그것을 씀 (동일성 확인: 에이전트는 train 전체로 정하고 PCA 만 fold 의 train 으로 fit)."""
    A, B = A.astype(np.float32), B.astype(np.float32)
    isbin = binary_cols(A) if isbin is None else isbin
    pca = PCA(N_PCA, random_state=0).fit(A)

    def f(M):
        C = M[:, ~isbin]
        return np.hstack([pca.transform(M), M[:, isbin].sum(1, keepdims=True), C.mean(1, keepdims=True),
                          C.std(1, keepdims=True)]).astype(np.float32)

    names = [f"pca_{i}" for i in range(N_PCA)] + ["bin_sum", "cont_mean", "cont_std"]
    return f(A), f(B), names


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    a, b, names = blocks(X.to_numpy(), X_test.to_numpy())
    return (pd.concat([X, pd.DataFrame(a, columns=names, index=X.index)], axis=1),
            pd.concat([X_test, pd.DataFrame(b, columns=names, index=X_test.index)], axis=1), cats)
