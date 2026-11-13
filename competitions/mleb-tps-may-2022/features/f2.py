"""C1 의 F2: MLE-bench 에이전트 Codex × GPT 풀이의 타깃 비의존 피처 전부, 244열 (docs/plans/roadmap.md).

출처: 실행 tabular-playground-series-may-2022__codex-gpt__r1__20261112-102453 (test 0.99734) 의 작업 디렉터리.
최종 제출 (build_submission2.py) = NN 10개 (nn_experiment·nn_gmm·nn_gmm2·nn_gmmlog, 전체 재학습 nn_full) + LightGBM 2개 (lgb2·lgb4, 재학습 lgb_full)
의 온도 섞기 + meta_correction. 피처는 그 성분들이 쓴 것의 합집합 → lgb4.py 의 193열 (xgbgmm.log 의 "data (900000, 193)") 에
nn_gmmlog.py 만 쓴 gmm_log 51열을 더함. 계산식·상수·순서·float32 변환은 에이전트 그대로:
  원본      f_27 을 뺀 30열 (F0)
  f_27      글자 위치별 10열 (A=0), 글자별 개수 20열, 서로 다른 글자 수, 글자 코드 합, 이웃 글자가 같은 수   (lgb2.py)
  f_28      f_00..f_06 으로 선형 회귀한 나머지 (train+test 전체로 fit)                                   (lgb2.py)
  pca       f_19..f_26 의 PCA 8개 (train+test 전체로 fit)                                                (lgb2.py)
  쌍        f_00·f_01·f_02·f_05 의 두 개씩 곱·합·차 18열, 글자 위치 두 개씩 같은지 45열                     (lgb4.py)
  gmm       f_30 값마다 f_19..f_26 (표준화) 에 GaussianMixture (성분 2·3·4·6·8·12·16) → 성분 확률 + 로그 가능도, 58열
            (cluster_features.py: 4·8·16, cluster_features2.py: 2·3·6·12, 각자 rng(42) 로 fit 행을 뽑음)
  gmm_log   gmm 성분 확률의 log (1e-15 에서 자름), 51열                                                    (Codex 셸 명령)
빼 둔 것: stacked.npy (stacking_features.py, OOF 예측 = 타깃 사용, 최종 성분은 안 씀), meta_correction (검증 예측 위에서 학습한
LightGBM 보정 = 모델 쪽), NN 의 원-핫·StandardScaler (모델 입력 인코딩), lgb 의 f_27 글자 열 범주형 지정 (F1 과 같게 수치형).

GMM 은 로컬 8코어 약 30초 (에이전트 기록 약 2.5분) → data_dir/_features/ 에 캐시 (fab.core.features.cached: 파일 잠금, 모델별 프로세스가 동시에 시작해도 한 번만 계산).
TFM 피처 상한 없음 (244열).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module
from fab.core.features import cached

_f1 = load_module(Path(__file__).with_name("f1.py"), "features_tps_may_2022_f1")

TFM_MAX_FEATURES = None
CACHE_VERSION = 1  # GMM 계산을 바꾸면 올림
GMM_RUNS = [  # (성분 수 목록, fit 행 상한, n_init): 에이전트 스크립트 하나 = rng 하나
    ([4, 8, 16], 80000, 1),  # cluster_features.py
    ([2, 3, 6, 12], 150000, 2),  # cluster_features2.py
]
GMM_ORDER = [2, 3, 4, 6, 8, 12, 16]  # lgb4.py·nn_gmm2.py 가 붙인 순서


def gmm_features(data: pd.DataFrame) -> dict[int, np.ndarray]:
    """에이전트 cluster_features.py·cluster_features2.py 그대로: n → (행, n+1) float32 (성분 확률 n개 + score_samples)."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(data[[f"f_{i:02}" for i in range(19, 27)]].to_numpy())
    classes = data["f_30"].to_numpy()
    out = {}
    for ns, max_fit, n_init in GMM_RUNS:
        rng = np.random.default_rng(42)
        for n in ns:
            f = np.zeros((len(data), n + 1), dtype=np.float32)
            for cat in np.unique(classes):
                ids = np.flatnonzero(classes == cat)
                fit = rng.choice(ids, min(len(ids), max_fit), replace=False)
                model = GaussianMixture(n_components=n, covariance_type="full", reg_covar=0.001, max_iter=75,
                                        n_init=n_init, random_state=42, tol=0.001)
                model.fit(X[fit])
                f[ids, :n] = model.predict_proba(X[ids])
                f[ids, n] = model.score_samples(X[ids])
            out[n] = f
    return out


def build(train: pd.DataFrame, test: pd.DataFrame, data_dir: Path):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    ids = np.r_[train["id"].to_numpy(), test["id"].to_numpy()]
    data = pd.concat([train.drop(columns="target"), test], ignore_index=True).drop(columns="id")
    chars = _f1.chars(data.pop("f_27")).astype(np.uint8)  # 에이전트와 같은 uint8 (f1.chars 는 길이 10 확인)
    new = {}
    for i in range(10):
        new[f"f_27_{i}"] = chars[:, i]
    for i in range(20):
        new[f"f_27_count_{i}"] = (chars == i).sum(axis=1).astype("uint8")
    new["f_27_unique"] = sum((new[f"f_27_count_{i}"] > 0) for i in range(20)).astype("uint8")
    new["f_27_sum"] = chars.sum(axis=1).astype("uint8")
    new["f_27_repeat"] = (chars[:, :-1] == chars[:, 1:]).sum(axis=1).astype("uint8")
    nums = data[[f"f_{i:02}" for i in range(7)]].to_numpy()
    lin = LinearRegression().fit(nums, data["f_28"])
    new["f_28_resid"] = data["f_28"] - lin.predict(nums)
    pcs = PCA(n_components=8).fit_transform(data[[f"f_{i:02}" for i in range(19, 27)]])
    for i in range(8):
        new[f"pca_{i}"] = pcs[:, i]
    for a, b in combinations([0, 1, 2, 5], 2):
        left, right = data[f"f_{a:02}"], data[f"f_{b:02}"]
        new[f"prod_{a}_{b}"] = left * right
        new[f"sum_{a}_{b}"] = left + right
        new[f"diff_{a}_{b}"] = left - right
    for a, b in combinations(range(10), 2):
        new[f"char_equal_{a}_{b}"] = (chars[:, a] == chars[:, b]).astype("uint8")
    gmm = cached(f"tps_gmm_v{CACHE_VERSION}", ids, data_dir,
                 lambda: {f"gmm{n}": v for n, v in gmm_features(data).items()})
    gmm = {n: gmm[f"gmm{n}"] for n in GMM_ORDER}
    for n in GMM_ORDER:
        for i in range(n + 1):
            new[f"gmm{n}_{i}"] = gmm[n][:, i]
    for n in GMM_ORDER:  # nn_gmmlog.py 의 gmm_log.npy: 성분 확률만 (score_samples 열 빼고)
        q = np.log(np.clip(gmm[n][:, :n], 1e-15, 1))
        for i in range(n):
            new[f"gmm_log{n}_{i}"] = q[:, i]
    data = pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)
    data = data.astype({c: "float32" for c in data.select_dtypes("float64").columns})
    n = len(train)
    return data.iloc[:n].reset_index(drop=True), data.iloc[n:].reset_index(drop=True), []
