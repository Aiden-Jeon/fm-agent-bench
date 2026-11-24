"""fev 0.10.0 의 예측 지표 (SQL · MASE · WQL · WAPE) 를 numpy 로 옮긴 것 (원격 정리 작업 환경에 fev 가 없어서, docs/forecast.md 4).

단변량 window 하나: 시리즈 N 개 × horizon H. fev `metrics.py` 와 같은 계산 순서:

- 계절 오차 s_i = nanmean_t |y_t − y_{t−m}| (과거 전체, m = seasonality). 과거가 m 이하이거나 차가 모두 NaN 이면 NaN
- MASE = mean_{i,h} |y − ŷ| / s_i,  SQL = mean_τ mean_{i,h} QL_τ / s_i  (QL_τ = 2·|(y − q̂_τ)(1[y ≤ q̂_τ] − τ)|)
  → NaN·±inf (s_i 가 0·NaN 인 시리즈, 정답 NaN) 은 평균에서 뺌 (fev 의 _safemean)
- WAPE = nanmean |y − ŷ| / max(nanmean |y|, ε),  WQL = mean_τ nanmean QL_τ / max(nanmean |y|, ε)  (fev-bench 는 ε = 1)

fev 와 같은 값인지는 로컬에서 driver.py score 로 확인 (fab fc check). 입력 CSV 가 float32 값을 짧은 글자로 적은 것이라
fev (float32 → float64) 와 상대 1e−6 정도 다를 수 있음.
"""

from __future__ import annotations

import warnings

import numpy as np

WAPE_EPS = WQL_EPS = 1.0  # fev-bench 의 extra_metrics 설정


def _safemean(a: np.ndarray, axis=None):
    a = np.where(np.isfinite(a), a, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def seasonal_error(past: list[np.ndarray], m: int) -> np.ndarray:
    """시리즈마다 과거 값 배열 → s_i (N,)."""
    out = np.full(len(past), np.nan)
    for i, y in enumerate(past):
        y = np.asarray(y, dtype=np.float64)
        if len(y) > m:
            d = np.abs(y[m:] - y[:-m])
            if (~np.isnan(d)).any():
                out[i] = np.nanmean(d)
    return out


def quantile_loss(y: np.ndarray, q: np.ndarray, levels: list[float]) -> np.ndarray:
    """y (N, H), q (N, H, Q) → (N, H, Q)."""
    ye = y[..., None]
    return 2 * np.abs((ye - q) * ((ye <= q) - np.asarray(levels)))


def scores(y: np.ndarray, point: np.ndarray, q: np.ndarray, levels: list[float], s: np.ndarray) -> dict[str, float]:
    """y·point (N, H), q (N, H, Q), s (N,) → {SQL, MASE, WQL, WAPE} (fev 와 같은 이름)."""
    y, point, q, s = (np.asarray(a, dtype=np.float64) for a in (y, point, q, s))
    with np.errstate(divide="ignore", invalid="ignore"):
        ql = quantile_loss(y, q, levels)
        sql = float(np.mean(_safemean(ql / s[:, None, None], axis=(0, 1))))
        mase = float(_safemean(np.abs(y - point) / s[:, None]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            abs_true = np.nanmean(np.abs(y))
            wape = float(np.nanmean(np.abs(y - point)) / max(abs_true, WAPE_EPS))
            wql = float(np.mean(np.nanmean(ql, axis=(0, 1)) / max(abs_true, WQL_EPS)))
    return dict(SQL=sql, MASE=mase, WQL=wql, WAPE=wape)
