"""C1 의 F2: TabArena Agent AIDE × GPT 풀이의 백색화 30열을 원본 30열에 더한 60열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p01__aide-gpt__r1__20261114-090001 (파일럿 1회째, r0f0 test 1 − AUC 0.01624) 의 최종 풀이
(workspace/logs/exp/best_solution.py = journal best 노드 37b9f7bf, step 35). 카드: competitions/ta-hazelnut-spread-contaminant-detection/c1.md
(PR #34). hazelnut 의 Agent 최고 Codex × GPT (0.00640) 는 R3(b) 로 뺌: 최종 finalize_submission.py 의 성분이 모두 스스로 깐 TabICL 에
원본 열 (+ 잡음 늘리기 · 의사 레이블 · 순위 섞기) 이라 피처 블록이 없음 → 다음 순위 AIDE × GPT (PR #34 의 "고르는 규칙").
최종 제출 = Matérn 꼴 커널 SVC (거리 = 백색화 공간의 유클리드 + 공분산 대비 metric + Fisher 판별 방향) 5설정 중 dev CV 상위 2개 평균,
전체 fit 과 5-fold 평균을 반반. 옮긴 것 (계산식은 에이전트 그대로, best_solution.py:91-106·153-154 의 SensorMetric):
  z_0~z_29   StandardScaler → 평균 빼기 → 공분산 (n − 1 로 나눔) 의 eigh → whitening = 고유벡터 / √(max(λ, 0) + 0.1),
             Z = 중심화한 값 @ whitening. 에이전트는 fold 마다 학습 행으로 fit 하고 나머지는 transform → 여기서는 train 으로 fit,
             test 는 transform. 모델 바로 앞의 전처리도 블록으로 봄 (PR #34 의 R3 해석, 2026-11-18 사용자)
열: 원본 30 (F0) + Z 30 = 60. 카드의 ~f2s 는 Z 30열만이지만 다른 f2.py 처럼 F0 에 블록을 더하는 꼴로 둠 (2026-11-18 사용자.
카드 처음 판은 원본을 31열로 적었고, #34 머지 전에 30열로 고침).
뺀 것: Fisher 판별 방향 · 공분산 대비 metric · Ledoit–Wolf 수축 (y 로 fit = 타깃 의존, 모델 쪽), 커널 · SVC · 설정 고르기 · fold 평균
(모델), 제출 점수의 z-score·시그모이드 (단조, AUC 무관).
train+test fit: 없음 (에이전트도 학습 행으로만 fit).
동일성 확인 (로컬, 2026-11-18): train 전체로 fit 한 Z 가 에이전트 SensorMetric().fit_transform 과 같은지, 이 함수의 Z 로 바꾼
best_solution.py 를 r0f0 데이터로 돌려 카드의 dev 5-fold AUC 0.985474 · holdout AUC 0.988203 이 나오는지 (docs/tabarena.md 의 "C1").
TFM 피처 상한 없음 (60열).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_hazelnut_f0")

TFM_MAX_FEATURES = None


class Whitening:
    """에이전트 SensorMetric 의 표준화 · 백색화 부분 (y 를 쓰는 부분은 뺌)."""

    def fit(self, X: np.ndarray) -> "Whitening":
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        standardized = self.scaler.fit_transform(X)
        self.mean = standardized.mean(axis=0)
        centered = standardized - self.mean
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        self.whitening = vectors / np.sqrt(np.maximum(values, 0.0) + 0.1)[None, :]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self.scaler.transform(X) - self.mean) @ self.whitening


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    cols = [c for c in test.columns if c != "id"]  # 에이전트의 features (id·타깃 뺀 30열)
    w = Whitening().fit(train[cols].to_numpy(np.float64))
    names = [f"z_{i}" for i in range(len(cols))]
    z_tr = pd.DataFrame(w.transform(train[cols].to_numpy(np.float64)), columns=names)
    z_te = pd.DataFrame(w.transform(test[cols].to_numpy(np.float64)), columns=names)
    return pd.concat([X, z_tr], axis=1), pd.concat([X_test, z_te], axis=1), cats
