"""평가 지표. 분류는 확률 행렬 (n, n_classes), 회귀는 예측값 (n,) 을 받음."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

PROBA_NEG_TOL = 1e-6  # 이보다 작은 음수는 수치 오차로 보고 0 으로 자름. 더 크면 확률이 아님 (래퍼 버그)


def normalize_proba(p: np.ndarray) -> np.ndarray:
    """분류 확률 (n, n_classes) 의 행 합을 1 로 맞춤. TFM 은 float32·float16 확률이라 99 클래스(leaf)면 합이 1 에서
    눈에 띄게 벗어남 → sklearn 1.9 log_loss 는 경고만 하고 그대로 계산, MLE-bench leaf 채점기는 1 ± 1e-6 이 아니면 무효.
    NaN·inf, 뚜렷한 음수, 합이 0 인 행은 고치지 않고 에러 (확률이 아닌 출력을 그럴듯한 제출로 바꾸지 않게)."""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError(f"확률에 NaN/inf 가 있음 ({int((~np.isfinite(p)).any(1).sum())}행)")
    if (p < -PROBA_NEG_TOL).any():
        raise ValueError(f"확률에 음수가 있음 (최솟값 {p.min():.3g}) → 확률이 아닌 출력")
    p = np.clip(p, 0, None)
    total = p.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError(f"확률 합이 0 인 행 {int((total <= 0).sum())}개")
    return p / total


def _accuracy(y, p):
    return skm.accuracy_score(y, p.argmax(1))


def _auc(y, p):
    return skm.roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else skm.roc_auc_score(y, p, multi_class="ovr")


METRICS = {
    # name: (fn, higher_is_better)
    "accuracy": (_accuracy, True),
    "auc": (_auc, True),
    "logloss": (lambda y, p: skm.log_loss(y, p, labels=np.arange(p.shape[1])), False),
    "rmse": (lambda y, p: skm.root_mean_squared_error(y, p), False),
    "mae": (lambda y, p: skm.mean_absolute_error(y, p), False),
    "r2": (lambda y, p: skm.r2_score(y, p), True),
}

# 대회 지표 외에 같이 기록할 보조 지표
EXTRA = {"binary": ["accuracy", "auc", "logloss"], "multiclass": ["accuracy", "logloss"], "regression": ["rmse", "mae"]}


def score(name: str, y: np.ndarray, pred: np.ndarray) -> float:
    return float(METRICS[name][0](y, pred))
