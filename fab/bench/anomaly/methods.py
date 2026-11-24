"""이상탐지 방법: pyod 기본 설정 (CPU) 과 TabPFN 비지도 확장 (GPU). fit(X_train) → score(X_test) (클수록 이상).

이름 뒤 `+clean` = 깨끗한 context 진단 (docs/anomaly.md 결정 6): train 중 라벨이 정상 (0) 인 행만으로 fit. train 라벨을 쓰므로
비지도가 아님 → Elo 에 안 넣음. 원격에서만 (train 라벨은 FM 작업만 읽는 내 Volume 자리, run.py).

무거운 import (pyod · torch · tabpfn) 는 함수 안 (로컬 제출 명령이 이 파일을 읽어도 설치가 필요 없게).
"""

from __future__ import annotations

import os
import random

import numpy as np

CLEAN = "+clean"
# 이름: (pyod 모듈, 클래스). ADBench 의 비지도 14개 중 넣은 10개 (docs/anomaly.md 3). 모두 인자 없이 (pyod 기본)
PYOD = {
    "iforest": ("pyod.models.iforest", "IForest"),
    "pca": ("pyod.models.pca", "PCA"),
    "ocsvm": ("pyod.models.ocsvm", "OCSVM"),
    "lof": ("pyod.models.lof", "LOF"),
    "cblof": ("pyod.models.cblof", "CBLOF"),
    "hbos": ("pyod.models.hbos", "HBOS"),
    "knn": ("pyod.models.knn", "KNN"),
    "copod": ("pyod.models.copod", "COPOD"),
    "ecod": ("pyod.models.ecod", "ECOD"),
    "loda": ("pyod.models.loda", "LODA"),
}
ANCHOR = "iforest"  # Elo 기준점 = 1000 (docs/anomaly.md 결정 7)
# TabPFN 비지도 확장 (tabpfn-extensions TabPFNUnsupervisedModel.outliers). version = TabPFN 가중치 (결정 2: 패키지 기본 v3.5)
FM = {"tabpfn_unsup": dict(version="v3.5", n_permutations=10)}
PYOD_PIP = [["pyod==3.6.8"]]  # 로컬 재현 확인 (docs/anomaly.md 2) 과 같은 버전
FM_PIP = [["tabpfn==9.2.0"], ["tabpfn-extensions==0.6.3"]]  # tabpfn 은 FM 실행 이미지 (docker/air) 와 같은 버전


def base(name: str) -> str:
    return name.removesuffix(CLEAN)


def is_clean(name: str) -> bool:
    return name.endswith(CLEAN)


def is_known(name: str) -> bool:
    return base(name) in PYOD or base(name) in FM


def is_gpu(name: str) -> bool:
    return base(name) in FM


def remote_pip(names: list[str]) -> list[list[str]]:
    out = []
    for n in names:
        for a in (FM_PIP if is_gpu(n) else PYOD_PIP if base(n) in PYOD else []):
            if a not in out:
                out.append(a)
    return out


def opts(name: str) -> str:
    """결과표의 opts 열: 그 방법의 설정 (pyod 는 기본 = 'default', FM 은 가중치 버전 · 순열 수)."""
    b = base(name)
    s = " ".join(f"{k}={v}" for k, v in sorted(FM[b].items())) if b in FM else "default"
    return f"{s} context=clean" if is_clean(name) else s


def set_seed(seed: int):
    """ADBench 의 Utils.set_seed 와 같은 자리 (numpy · random 전역, torch 는 있으면). pyod 의 random_state 기본값은 None 이라
    전역 numpy 상태로 정해짐 → ADBench 와 같은 값 (docs/anomaly.md 2 의 재현 확인)."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def pyod_scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    import importlib

    mod, cls = PYOD[base(name)]
    set_seed(seed)
    model = getattr(importlib.import_module(mod), cls)().fit(X_train)
    return np.asarray(model.decision_function(X_test), dtype=float)


def tabpfn_scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    """−log 밀도 (tabpfn-extensions 의 outliers 는 낮을수록 이상 → 부호를 뒤집음). 가중치는 FM[...]['version']
    (v3.5 는 받을 때 라이선스 동의가 필요 → run.py 가 TABPFN_TOKEN 을 원격 secret 에서 넣음)."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # 동의가 안 돼 있으면 브라우저를 기다리지 말고 바로 실패
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    cfg = FM[base(name)]
    version = ModelVersion(cfg["version"])
    set_seed(seed)
    model = TabPFNUnsupervisedModel(tabpfn_clf=TabPFNClassifier.create_default_for_version(version),
                                    tabpfn_reg=TabPFNRegressor.create_default_for_version(version))
    model.fit(torch.tensor(X_train, dtype=torch.float32))
    set_seed(seed)  # 순열 (python random) 을 seed 로
    log_density = model.outliers(torch.tensor(X_test, dtype=torch.float32), n_permutations=cfg["n_permutations"])
    return -np.asarray(log_density.detach().cpu().numpy() if hasattr(log_density, "detach") else log_density, dtype=float)


def scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    fn = tabpfn_scores if is_gpu(name) else pyod_scores
    s = fn(name, X_train, X_test, seed)
    if s.shape != (len(X_test),) or not np.isfinite(s).all():
        raise ValueError(f"{name}: 점수 모양 {s.shape} 또는 NaN/inf {int((~np.isfinite(s)).sum())}개")
    return s
