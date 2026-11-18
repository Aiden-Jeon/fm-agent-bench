"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 무작위 사영 32열을 더한 38열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p05__codex-gpt__r1__20261116-133105 (r0f0 test RMSE 0.9003, QSAR_fish 의 Agent 최고) 의 작업 디렉터리.
최종 제출 = 셸에서 이전 제출에 kNN 잔차 보정을 여러 번 더한 것. 그 바탕은 mix_submission.py · residual_submission.py ·
best_submission.py 의 섞기 (.4 forest · .25 kNN · .2 스스로 깐 TabICL · .15 SVR 등). 피처를 받는 표형 성분은 forest
(refined_submission.py·mix_submission.py 의 ExtraTrees) 이고, 입력은 power_features 의 출력:
  열 3·4 (NdsCH·NdssC) 에 sqrt → 열 0·1·2·5 에 Yeo-Johnson (PowerTransformer(standardize=False)) → 6열 표준화 → 6 × 32 정규 난수
  (RandomState(42), 열마다 길이 1) 로 사영. 변환·표준화는 train 만으로 fit.
  → 표준화한 6열 + 사영 32열. 표준화한 6열은 원본의 단조 변환이라 원본 6열 (F0) 을 그대로 두고 사영 32열 (proj_0…31) 만 더함.
forest 는 log1p 변형 (seed 42·11·19·29·43) 도 섞지만, 가중치가 가장 큰 sqrt · seed 42 (0.5) 하나만 옮김.
빼 둔 것: SVR (입력의 expm1(열 1) 포함)·가중 kNN·라플라스 커널 SVR·ExtraTrees·TabICL (모델), log1p 변형의 사영 (같은 꼴의 다른 seed),
train 과 같은 행에 train 정답 섞기 · kNN 잔차 보정 · 상수 보정 (train 정답을 쓰는 후처리), 섞기. 타깃 피처 없음.
train+test 로 fit 한 것: 없음 (PowerTransformer·StandardScaler 는 train 만, 사영 행렬은 고정 seed) → inductive.
동일성 확인 (로컬, 2026-11-18): 이 사영으로 refined_submission.py 의 forest 를 다시 계산해 refined_predictions.npz 의 forest 와
원소별 비교 (docs/tabarena.md 의 "C1"). TFM 피처 상한 없음.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_qsar_fish_f0")

TFM_MAX_FEATURES = None


def power_features(a: np.ndarray, b: np.ndarray, seed: int, count: str) -> tuple[np.ndarray, np.ndarray]:
    """에이전트 refined_submission.py 의 power_features 그대로 (a = train, b = test, 열은 test.csv 순서).
    돌려주는 것은 (표준화 6열 + 사영 32열) 의 train·test."""
    from sklearn.preprocessing import PowerTransformer, StandardScaler

    a = a.copy()
    b = b.copy()
    f = np.log1p if count == "log" else np.sqrt
    a[:, 3:5] = f(a[:, 3:5])
    b[:, 3:5] = f(b[:, 3:5])
    ids = [0, 1, 2, 5]
    tf = PowerTransformer(standardize=False).fit(a[:, ids])
    a[:, ids] = tf.transform(a[:, ids])
    b[:, ids] = tf.transform(b[:, ids])
    sc = StandardScaler().fit(a)
    a = sc.transform(a)
    b = sc.transform(b)
    rng = np.random.RandomState(seed)
    w = rng.normal(size=(6, 32))
    w = w / np.linalg.norm(w, axis=0)
    return np.concatenate([a, a @ w], axis=1), np.concatenate([b, b @ w], axis=1)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    cols = [c for c in test.columns if c != "id"]
    a, b = power_features(train[cols].to_numpy(), test[cols].to_numpy(), 42, "sqrt")
    names = [f"proj_{i}" for i in range(32)]
    return (pd.concat([X, pd.DataFrame(a[:, 6:], columns=names)], axis=1),
            pd.concat([X_test, pd.DataFrame(b[:, 6:], columns=names)], axis=1), cats)
