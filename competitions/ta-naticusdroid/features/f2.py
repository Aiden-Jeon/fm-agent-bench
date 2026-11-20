"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 개수 · 꼴 빈도 · 해밍 이웃 수 7열 (train 으로만 다시 fit) 을 더한 93열 (docs/tabarena.md 의 "행 추가: NATICUSdroid").

출처: 실행 tabarena-p17__aide-sonnet__r1__20261120-225541 (r0f0 test 1 − AUC 0.01123) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 7a5a8ef7, "ball" 판, 에이전트 OOF AUC 0.98887). 카드: competitions/ta-naticusdroid/c1.md.
이 행의 Agent 최고는 Codex × GPT (0.00855) 지만 FM 보다 나은 몫이 test 행 배정 후처리에서 나와 R3 (a) 로 내려옴 (카드).
최종 제출 = CatBoost (depth 8, rsm 0.15) 5-fold × seed 3 의 test 예측 평균. 옮긴 것 (계산식은 에이전트 그대로):
  개수     n_perm (요청한 권한 수), n_sens (이름에 SMS · PHONE · LOCATION 등 10개 말이 든 권한 수),
           n_vendor (android.permission. 로 시작하지 않는 권한 수)  best_solution.py:15-38, 행 단위
  꼴 빈도  pfreq: 86열 0/1 꼴이 같은 행의 수 (자기 포함)  :41-47
  이웃 수  nn1 · nn2 · nn4: 해밍 거리 1 · 2 · 4 안의 다른 행 수  :49-62
뺀 것: 꼴 타깃 인코딩 pte · pcnt (:76-93 · :111-114, 타깃 의존), 해밍 이웃 레이블 평균 b2_te · b4_te 와 그 개수 b2_c · b4_c (:95-142, 타깃 의존,
개수는 같은 계산을 fold 안에서 한 것이라 아래 train 판 nn2 · nn4 와 겹침), CatBoost · seed 평균 · base/ball 고르기.
train+test fit: 에이전트는 pfreq · nn 을 train+test 로 셈 → 같은 계산을 train 행으로만 fit (R3 해석, 2026-11-18 사용자).
train 판: pfreq = 자기 + train 의 같은 꼴 수 (test 행도 자기를 셈, 그래서 train 에 없는 꼴은 1), nn = train 행 중 거리 안의 다른 행 수.
동일성 확인 (로컬): with_test=True 판 + 에이전트 식의 pte · pcnt 로 에이전트 "base" 판 CatBoost (같은 설정 · fold · seed) 를
우리 코드로 다시 돌려 기록 OOF AUC 0.9886678703540603 과 견준 뒤 train 판으로 바꿈 (docs/tabarena.md). TFM 피처 상한 없음 (93열).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_naticusdroid_f0")

TFM_MAX_FEATURES = None
SENS_KEYS = ["SMS", "PHONE", "LOCATION", "CONTACTS", "CALL", "RECORD_AUDIO", "CAMERA", "INSTALL", "DEVICE_ADMIN", "SYSTEM_ALERT"]
RADII = {"nn1": 1, "nn2": 2, "nn4": 4}


def blocks(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], with_test: bool = False):
    """에이전트 식의 7열. with_test=True 면 에이전트처럼 pfreq · nn 을 train+test 로 셈 (동일성 확인용)."""
    A_tr, A_te = train[feats].astype(int).to_numpy(), test[feats].astype(int).to_numpy()
    sens = [c for c in feats if any(k in c for k in SENS_KEYS)]
    vendor = [c for c in feats if not c.startswith("android.permission.")]
    out = []
    for df, A in ((train, A_tr), (test, A_te)):
        X = pd.DataFrame(index=range(len(df)))
        X["n_perm"] = A.sum(1).astype(float)
        X["n_sens"] = df[sens].astype(int).sum(axis=1).to_numpy(float)
        X["n_vendor"] = df[vendor].astype(int).sum(axis=1).to_numpy(float)
        out.append(X)
    pat_tr = pd.util.hash_pandas_object(train[feats].astype(int), index=False).to_numpy()
    pat_te = pd.util.hash_pandas_object(test[feats].astype(int), index=False).to_numpy()
    vc = pd.Series(np.concatenate([pat_tr, pat_te]) if with_test else pat_tr).value_counts()
    out[0]["pfreq"] = pd.Series(pat_tr).map(vc).fillna(0).to_numpy(float)
    out[1]["pfreq"] = pd.Series(pat_te).map(vc).fillna(0).to_numpy(float) + (0 if with_test else 1)  # train 판: 자기 + train 의 같은 꼴
    ref = np.vstack([A_tr, A_te]) if with_test else A_tr  # 이웃을 세는 행
    R = ref.astype(np.float32)
    for X, A, is_ref in ((out[0], A_tr, True), (out[1], A_te, with_test)):
        Q = A.astype(np.float32)
        ham = Q.sum(1)[:, None] + R.sum(1)[None, :] - 2 * (Q @ R.T)
        for name, r in RADII.items():
            X[name] = ((ham <= r).sum(1) - (1 if is_ref else 0)).astype(float)  # 자기 행은 뺌 (에이전트의 −1)
    return out[0], out[1]


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    feats = list(X.columns)
    b_tr, b_te = blocks(train, test, feats)
    return pd.concat([X, b_tr], axis=1), pd.concat([X_test, b_te], axis=1), cats
