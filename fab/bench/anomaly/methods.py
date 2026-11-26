"""이상탐지 방법: pyod 기본 설정 (CPU) 과 TabPFN 비지도 확장 (GPU). fit(X_train) → score(X_test) (클수록 이상).

이름 뒤 `+clean` = 깨끗한 context 진단 (docs/anomaly.md 결정 6): train 중 라벨이 정상 (0) 인 행만으로 fit. train 라벨을 쓰므로
비지도가 아님 → Elo 에 안 넣음. 원격에서만 (train 라벨은 FM 작업만 읽는 내 Volume 자리, run.py).
이름 뒤 `+trim<k>` · `+trim<k>x<n>` · `+iftrim<k>` = 라벨 없는 context 정리 (docs/anomaly.md "TFM 이 이득을 보는 조건" E3):
train 점수 상위 k% 를 context 에서 빼고 (n 번 반복) 남은 train 으로 fit. 점수는 그 방법 자신 (trim) 또는 IForest (iftrim).
공식 풀 밖의 변형 (`is_variant`) 은 채점 행이 results/anomaly_conditions.csv 로 감 (fab/bench/anomaly 의 grade).
`<모델>_clf+la<k>` = 라벨을 쓰는 분류기 (docs/anomaly.md "라벨이 있는 이상탐지"): train 이상 중 k% 에만 라벨 (ADBench la, 나머지는
정상 0 = PU) → fit(X_train, y) → P(이상). 채점 행은 results/anomaly_supervised.csv (`is_supervised`).
`fomo0d` = FoMo-0D (이상탐지 전용 PFN, 공개 체크포인트 그대로). 비지도 → 공식 풀 (2026-11-26 사용자, 결정 19).

무거운 import (pyod · torch · tabpfn) 는 함수 안 (로컬 제출 명령이 이 파일을 읽어도 설치가 필요 없게).
"""

from __future__ import annotations

import os
import random
import re

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
# TabPFN 비지도 확장 (tabpfn-extensions TabPFNUnsupervisedModel.outliers). version = TabPFN 가중치 (결정 2: v2.
# 패키지 기본 v3.5 는 쓰지 않음 (2026-11-25 사용자). v2 는 Prior Labs License 1.1, 받을 때 동의 없음, 우리 tabpfn_v2 와 같은 가중치)
# tabpfn_unsup_p3 = 순열 3 (조건 실험 E0: 순열 10 과 같은지 보고, 같으면 조건 실험의 TabPFN 으로). 공식 풀은 tabpfn_unsup 만
# kumo_unsup = Kumo (sdm 0.2.1, KumoTabular) 로 같은 chain rule 을 만든 **우리 래퍼** (sdm 에 이상탐지 · 밀도 API 가 없음, 2026-11-25
# 휠 소스 확인). 연속 열은 999 분위수 → CDF (분위수 사이 선형, 바깥 0.1% 는 지수 꼬리) → 폭 2h 구간 확률 / 2h (h, MinMax 단위).
# docs/anomaly.md "Kumo"
FM = {"tabpfn_unsup": dict(version="v2", n_permutations=10),
      "tabpfn_unsup_p3": dict(version="v2", n_permutations=3),
      "kumo_unsup": dict(model="kumo", n_permutations=10, n_estimators=8, h=0.005)}
# FoMo-0D (Shen · Wen · Akoglu, TMLR 2025, MIT): 저장소 코드 (모델 정의 · 전처리) 와 HF 가중치를 고정 커밋으로. 피처 100 (적으면 0 채움),
# context ≤ 4,999 행 (원문 inf_len 5000), 전처리는 원문 pfn_inference_transform('quantile') 을 train 으로만 정규화 (결정 17)
FM["fomo0d"] = dict(model="fomo0d", code="34d4339", weights="d0638ef", max_context=4999, transform="quantile")
FOMO_REPO, FOMO_SHA = "A-Chicharito-S/FoMo-0D", "34d4339e318d96ac744850188a547b62387a718c"
FOMO_HF, FOMO_REV = "YuchenShen/FoMo-0D", "d0638efa3cc7ec93ffcc9f9aa2ed6d15030e1161"
FOMO_PIP = [["einops==0.8.2"], ["safetensors==0.8.0"]]  # 이미지에 있으면 실행기가 건너뜀 (torch · huggingface_hub 는 있음)
# 공식 풀 (fm · agent) 의 참가자. 나머지 FM 이름 · 정리 변형은 조건 실험 표로. fomo0d 는 2026-11-26 사용자 결정으로 넣음 (결정 19)
OFFICIAL = [*PYOD, "tabpfn_unsup", "fomo0d"]
# 라벨을 쓰는 분류기: 이름 → (fab/models 의 모델 또는 ADBench 기본 GBDT, GPU 여부). +la<k> 가 꼭 붙음
CLF = {"tabpfn_clf": ("tabpfn_v2", True), "kumo_clf": ("kumo", True), "lgbm_clf": ("lgbm", False),
       "catboost_clf": ("catboost", False)}
LA = re.compile(r"\+la(\d+)$")
LA_LEVELS = [1, 5, 10, 25, 50, 100]  # % (ADBench rla_list 중 0.75 를 뺀 것, 작업 지시)
SUP_ANCHOR = "lgbm_clf"  # 지도 풀 기준점 = 1000 (la 마다, 결정 16)
TRIM = re.compile(r"\+(if)?trim(\d+)(?:x(\d+))?$")  # +trim10 · +trim10x2 · +iftrim10
PYOD_PIP = [["pyod==3.6.8"]]  # 로컬 재현 확인 (docs/anomaly.md 2) 과 같은 버전
FM_PIP = [["tabpfn==9.2.0"], ["tabpfn-extensions==0.6.3"]]  # tabpfn 은 FM 실행 이미지 (docker/air) 와 같은 버전
KUMO_PIP = [["structured-data-models==0.2.1"]]  # docker/air requirements 와 같은 버전 (이미지에 있으면 실행기가 건너뜀)
# tabpfn-extensions 0.6.3 의 범주 판정 (utils.infer_categorical_features · use_classifier_) 을 Kumo 래퍼도 그대로 씀
MAX_UNIQUE_CAT, MIN_ROWS_PER_CAT, MAX_CLASSES = 40, 10, 10


def base(name: str) -> str:
    return LA.sub("", TRIM.sub("", name.removesuffix(CLEAN)))


def la_spec(name: str) -> int | None:
    """+la<k> → k (%). 분류기가 아니면 None."""
    return int(m[1]) if (m := LA.search(name)) else None


def is_supervised(name: str) -> bool:
    """라벨을 쓰는 분류기 (+la<k>). 채점 행은 results/anomaly_supervised.csv."""
    return base(name) in CLF and la_spec(name) is not None


def is_fomo(name: str) -> bool:
    return FM.get(base(name), {}).get("model") == "fomo0d"


def is_clean(name: str) -> bool:
    return name.endswith(CLEAN)


def trim_spec(name: str) -> tuple[str, int, int] | None:
    """+trim<k>x<n> → (정리 점수를 내는 방법, k %, 반복 n). 정리 변형이 아니면 None."""
    if not (m := TRIM.search(name)):
        return None
    return ("iforest" if m[1] else base(name)), int(m[2]), int(m[3] or 1)


def is_variant(name: str) -> bool:
    """공식 풀 밖 (조건 실험): 순열을 바꾼 FM 이나 정리 변형. +clean 은 공식 대회에서 진단으로 이미 쓰는 것이라 변형이 아님."""
    return base(name) not in OFFICIAL or trim_spec(name) is not None


def is_known(name: str) -> bool:
    if base(name) in CLF:
        return la_spec(name) in LA_LEVELS and not is_clean(name) and trim_spec(name) is None
    return la_spec(name) is None and (base(name) in PYOD or base(name) in FM)


def is_gpu(name: str) -> bool:
    return base(name) in FM or CLF.get(base(name), (None, False))[1]


def is_kumo(name: str) -> bool:
    return FM.get(base(name), {}).get("model") == "kumo"


def remote_pip(names: list[str]) -> list[list[str]]:
    out = []
    for n in names:
        if base(n) in CLF:  # GBDT 는 CPU 작업 환경 · 이미지에 있음. TFM 분류기는 fab/models 와 같은 패키지
            need = {"tabpfn_clf": [["tabpfn==9.2.0"]], "kumo_clf": KUMO_PIP}.get(base(n), [])
        elif is_fomo(n):
            need = list(FOMO_PIP)
        else:
            need = ((KUMO_PIP + FM_PIP if is_kumo(n) else FM_PIP) if is_gpu(n) else []) + (PYOD_PIP if base(n) in PYOD else [])
        if (t := trim_spec(n)) and t[0] in PYOD:  # +iftrim: TabPFN 작업에서도 IForest 로 정리 점수
            need += PYOD_PIP
        for a in need:
            if a not in out:
                out.append(a)
    return out


def opts(name: str) -> str:
    """결과표의 opts 열: 그 방법의 설정 (pyod 는 기본 = 'default', FM 은 가중치 버전 · 순열 수)."""
    b = base(name)
    if b in CLF:
        return f"model={CLF[b][0]} la={la_spec(name) / 100:g}"
    s = " ".join(f"{k}={v}" for k, v in sorted(FM[b].items())) if b in FM else "default"
    if t := trim_spec(name):
        return f"{s} context=trim by={t[0]} k={t[1]} rounds={t[2]}"
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


def _fit_pyod(name: str, X: np.ndarray, seed: int):
    import importlib

    mod, cls = PYOD[base(name)]
    set_seed(seed)
    return getattr(importlib.import_module(mod), cls)().fit(X)


def pyod_scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(_fit_pyod(name, X_train, seed).decision_function(X_test), dtype=float)


def tabpfn_scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    """−log 밀도 (tabpfn-extensions 의 outliers 는 낮을수록 이상 → 부호를 뒤집음). 가중치는 FM[...]['version'] (v2:
    동의 없이 받음, 실행기의 모델 캐시에 남음). 동의가 필요한 버전으로 잘못 바꾸면 브라우저를 기다리지 말고 바로 실패."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    class Regressor(TabPFNRegressor):
        """tabpfn-extensions 는 y 를 logits 의 dtype 으로 맞춰 criterion (bar 분포) 에 넣는데, CUDA 에서는 bar 분포의 borders 가
        double · logits 가 float 이라 `y[mask] = borders[0]` 에서 dtype 오류 (2026-11-25 원격 첫 실행, H100. 로컬 MPS 는 둘 다
        float32 라 안 드러남) → full 출력의 logits 를 borders 의 dtype 으로 (log 밀도 계산이 double 이 될 뿐 값의 뜻은 같음)."""

        def predict(self, X, *args, output_type="mean", **kw):
            out = super().predict(X, *args, output_type=output_type, **kw)
            if output_type == "full" and torch.is_tensor(out.get("logits")):
                out["logits"] = out["logits"].to(dtype=out["criterion"].borders.dtype)
            return out

    cfg = FM[base(name)]
    version = ModelVersion(cfg["version"])
    set_seed(seed)
    model = TabPFNUnsupervisedModel(tabpfn_clf=TabPFNClassifier.create_default_for_version(version),
                                    tabpfn_reg=Regressor.create_default_for_version(version))
    model.fit(torch.tensor(X_train, dtype=torch.float32))
    set_seed(seed)  # 순열 (python random) 을 seed 로
    log_density = model.outliers(torch.tensor(X_test, dtype=torch.float32), n_permutations=cfg["n_permutations"])
    return -np.asarray(log_density.detach().cpu().numpy() if hasattr(log_density, "detach") else log_density, dtype=float)


def train_scores(name: str, X: np.ndarray, seed: int) -> np.ndarray:
    """context 정리용 train 행 점수 (클수록 이상). pyod 는 X 로 fit 한 in-sample 점수 (decision_scores_). TabPFN 은 행이 자기
    자신을 context 에 두면 밀도가 부풀므로 2겹: 무작위로 반을 나눠 서로를 context 로 점수."""
    if not is_gpu(name):
        return np.asarray(_fit_pyod(name, X, seed).decision_scores_, dtype=float)
    perm = np.random.default_rng([seed, len(X)]).permutation(len(X))
    a, b = perm[: len(X) // 2], perm[len(X) // 2:]
    s = np.empty(len(X))
    s[a] = scores(base(name), X[b], X[a], seed)
    s[b] = scores(base(name), X[a], X[b], seed)
    return s


def categorical_cols(X: np.ndarray) -> set[int]:
    """tabpfn-extensions 0.6.3 utils.infer_categorical_features 의 값 수 규칙: 값 ≤ 40 개 · 값마다 평균 10행 넘음."""
    out = set()
    for j in range(X.shape[1]):
        u = len(np.unique(X[:, j]))
        if 0 < u <= MAX_UNIQUE_CAT and len(X) / u > MIN_ROWS_PER_CAT:
            out.add(j)
    return out


def quantile_log_density(Q: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """행마다 분위수 (Q[i], 수준 0.001 … 0.999, 오름차순으로 정렬) 에서 y[i] 의 log 밀도.
    CDF: 분위수 사이는 선형, q001 아래 · q999 위는 질량 0.001 의 지수 꼬리 (척도 = 가장자리 10칸 폭 / 10 → 경계에서 밀도가
    이어짐, 최소 h). 밀도 = (F(y + h) − F(y − h)) / 2h (같은 값이 많아 분위수가 겹쳐도 유한). 꼬리 안에서는 닫힌 식 (log 로)."""
    Q = np.sort(Q, axis=1)
    n, m = Q.shape
    p = np.arange(1, m + 1) / (m + 1)
    lo, hi = Q[:, 0], Q[:, -1]
    s_lo = np.maximum((Q[:, 10] - lo) / 10, h)
    s_hi = np.maximum((hi - Q[:, -11]) / 10, h)
    tail = p[0]

    def cdf(v):
        inner = np.array([np.interp(v[i], Q[i], p) for i in range(n)])
        below = tail * np.exp(np.minimum(v - lo, 0) / s_lo)
        above = 1 - tail * np.exp(-np.maximum(v - hi, 0) / s_hi)
        return np.where(v < lo, below, np.where(v > hi, above, inner))

    with np.errstate(divide="ignore"):
        dens = np.log(np.maximum(cdf(y + h) - cdf(y - h), 1e-300)) - np.log(2 * h)
    # 구간이 통째로 꼬리 안: log(0.001) − 거리 / s + log(2 sinh(h / s)) − log 2h (뺄셈 없이)
    low = y + h < lo
    dens = np.where(low, np.log(tail) - (lo - y) / s_lo + np.log(2 * np.sinh(h / s_lo)) - np.log(2 * h), dens)
    up = y - h > hi
    dens = np.where(up, np.log(tail) - (y - hi) / s_hi + np.log(2 * np.sinh(h / s_hi)) - np.log(2 * h), dens)
    return dens


def kumo_scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int, chunk: int = 2048) -> np.ndarray:
    """Kumo 로 tabpfn-extensions outliers() 와 같은 chain rule (우리 래퍼, 공식 구현 아님). 점수 = −log 밀도 (순열 평균은 밀도의
    산술 평균 = log-sum-exp, 원문과 같음). 열 판정 · 첫 열의 잡음 입력 · 범주 값의 astype(int) 까지 tabpfn-extensions 0.6.3 그대로
    (같은 분해를 두 FM 이 쓰게). 범주 열 = 분류 (실제 클래스의 확률), 그 밖 = 회귀 (999 분위수 → quantile_log_density).
    회귀 타깃은 context 의 평균 · 표준편차로 표준화해 넘기고 분위수를 되돌림 (fab/models/tfm.py 의 _Transductive 와 같은 처리)."""
    import pandas as pd
    import sdm
    import torch
    from tabpfn_extensions.unsupervised.unsupervised import efficient_random_permutation  # 순열은 원문 함수 (같은 seed → 같은 순열)

    cfg = FM[base(name)]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = torch.amp.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    d = X_train.shape[1]
    cats = categorical_cols(X_train)

    def fit_predict(Xc, yc, Xq, classify):
        cols = [f"f{i}" for i in range(Xc.shape[1])]
        st = {c: "numerical" for c in cols}
        ctx = pd.DataFrame(Xc, columns=cols).assign(target=yc)
        ctx = sdm.TableTensor.from_pandas(ctx, stypes={**st, "target": "categorical" if classify else "numerical"},
                                          device=dev)
        model = sdm.models.KumoTabular(task="classification" if classify else "regression", device=dev)
        with autocast:
            model.fit(ctx.drop_columns("target"), ctx[:, "target"], num_estimators=cfg["n_estimators"])
        outs = []
        for i in range(0, len(Xq), chunk):
            q = sdm.TableTensor.from_pandas(pd.DataFrame(Xq[i:i + chunk], columns=cols), stypes=st, device=dev)
            with autocast:
                o = model.predict(q)
            o = o.to_pandas() if hasattr(o, "to_pandas") else pd.DataFrame(o.float().cpu().numpy())
            outs.append(o)
        return pd.concat(outs, ignore_index=True)

    set_seed(seed)
    logs = []
    for perm in efficient_random_permutation(list(range(d)), cfg["n_permutations"]):
        lp = np.zeros(len(X_test))
        for i, j in enumerate(perm):
            prev = list(perm[:i])
            if prev:
                Xc, Xq = X_train[:, prev], X_test[:, prev]
            else:  # 첫 열: 정규 잡음 한 열 (원문과 같음)
                Xc = torch.randn(len(X_train), 1).numpy()
                Xq = torch.randn(len(X_test), 1).numpy()
            yc, yq = X_train[:, j], X_test[:, j]
            if j in cats and len(np.unique(yc)) <= MAX_CLASSES:  # 원문의 use_classifier_ (값 수는 astype 전)
                yc_i, yq_i = yc.astype(int), yq.astype(int)
                out = fit_predict(Xc, yc_i, Xq, classify=True)
                classes = [int(float(c)) for c in out.columns]
                probs = out.to_numpy(float)
                col = {c: k for k, c in enumerate(classes)}
                pick = np.array([probs[r, col[v]] if v in col else 0.1 for r, v in enumerate(yq_i)])  # 원문: 모르는 값 0.1
                lp += np.log(np.maximum(pick, 1e-300))
            else:
                mu, sd = yc.mean(), yc.std() or 1.0
                out = fit_predict(Xc, (yc - mu) / sd, Xq, classify=False)
                Q = out[[f"q{k:03d}" for k in range(1, 1000)]].to_numpy(float) * sd + mu
                lp += quantile_log_density(Q, yq, cfg["h"])
        logs.append(lp)
    L = np.stack(logs)
    mx = L.max(0)
    return -(mx + np.log(np.exp(L - mx).mean(0)))


_FOMO = {}


def fomo_model():
    """FoMo-0D 모델 (한 프로세스에 한 번). 코드: FAB_FOMO0D (로컬: 받아 둔 저장소 폴더) 가 없으면 고정 커밋의 tarball 을 임시 폴더에
    받아 풂 (원격). 가중치: HF 고정 revision (실행기의 HF 캐시에 남음). 원문 pfns/layer.py 가 torch.nn.modules.transformer 에서
    Optional 을 import 하는데 요즘 torch 에 없음 → 그 이름만 채워 넣음 (원문 파일은 안 고침)."""
    if "model" in _FOMO:
        return _FOMO["model"], _FOMO["ft"]
    import sys
    import tarfile
    import tempfile
    import typing
    import urllib.request
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import torch.nn.modules.transformer as tt

    code = os.environ.get("FAB_FOMO0D")
    if not code:
        root = Path(tempfile.gettempdir()) / f"fomo0d-{FOMO_SHA[:7]}"
        code = root / f"FoMo-0D-{FOMO_SHA}"
        if not code.exists():
            root.mkdir(parents=True, exist_ok=True)
            tgz = root / "src.tgz"
            urllib.request.urlretrieve(f"https://codeload.github.com/{FOMO_REPO}/tar.gz/{FOMO_SHA}", tgz)
            with tarfile.open(tgz) as t:
                t.extractall(root, filter="data")
    if not hasattr(tt, "Optional"):
        tt.Optional = typing.Optional
    sys.path.insert(0, str(code))
    from data_prior.feature_transform import FeatureTransform
    from fomo_hub import FoMo0DHub

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FoMo0DHub.from_pretrained(FOMO_HF, revision=FOMO_REV, map_location=dev).to(dev).eval()
    gmm = SimpleNamespace(max_feature_dim=model.config["num_features"])
    _FOMO.update(model=model, ft=FeatureTransform(SimpleNamespace(prior=SimpleNamespace(gmm=gmm))))
    return _FOMO["model"], _FOMO["ft"]


def fomo_scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int, chunk: int = 1000) -> np.ndarray:
    """FoMo-0D 점수 = P(이상) (2 클래스 logit 의 softmax). context 가 상한 (4,999) 을 넘으면 seed 로 비복원 표본 (행 순서 유지, 원문은
    np.random.choice). 전처리는 원문 FeatureTransform.pfn_inference_transform (평균 · 표준편차 정규화 → 열마다 QuantileTransformer
    (normal) 를 context 로 fit → 크기 맞춤 · 0 채움) 을 normalize_with_test=False 로 (원문은 True = test 행도 정규화에 씀, 결정 17).
    질의 행은 묶어 넣음 (efficient eval masking 이라 질의끼리 안 봄: 하나씩 넣은 것과 최대 차 1e-6, 로컬 2026-11-26)."""
    import torch

    cfg = FM[base(name)]
    model, ft = fomo_model()
    dev = next(model.parameters()).device
    ctx = X_train
    if len(ctx) > cfg["max_context"]:
        ctx = ctx[np.sort(np.random.default_rng(seed).choice(len(ctx), cfg["max_context"], replace=False))]
    z = ft.pfn_inference_transform(np.concatenate([ctx, X_test]).astype(np.float64), cfg["transform"], len(ctx),
                                   normalize_with_test=False)
    tr = torch.tensor(z[:len(ctx)], dtype=torch.float32, device=dev).unsqueeze(1)
    te = torch.tensor(z[len(ctx):], dtype=torch.float32, device=dev)
    with torch.no_grad():
        out = torch.cat([model(tr, te[i:i + chunk].unsqueeze(1)) for i in range(0, len(te), chunk)]).squeeze(1)
    return out.float().softmax(-1)[:, 1].cpu().numpy().astype(float)


def clf_scores(name: str, X_train: np.ndarray, y: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    """라벨을 쓰는 분류기 → P(이상). y = PU 라벨 (라벨 준 이상 1, 나머지 0). TFM 은 fab/models 래퍼 (refit: train 전체로 한 번 fit,
    TabPFN v2 · Kumo 앙상블 8), GBDT 는 ADBench Supervised.py 와 같은 라이브러리 기본값 + random_state=seed (결정 15)."""
    import pandas as pd

    model, _ = CLF[base(name)]
    y = np.asarray(y).astype(int)
    if model == "lgbm":
        from fab.models.gbdt import import_omp

        clf = import_omp("lightgbm").LGBMClassifier(random_state=seed, verbose=-1)
    elif model == "catboost":
        from catboost import CatBoostClassifier

        clf = CatBoostClassifier(random_state=seed, verbose=0, allow_writing_files=False)
    else:
        from fab.models import make_model

        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        cols = [f"f{j}" for j in range(X_train.shape[1])]
        set_seed(seed)
        clf = make_model(model, "binary", [], seed)
        clf.fit(pd.DataFrame(X_train, columns=cols), y)
        return np.asarray(clf.predict_proba(pd.DataFrame(X_test, columns=cols)), dtype=float)[:, 1]
    clf.fit(X_train, y)
    return np.asarray(clf.predict_proba(X_test), dtype=float)[:, 1]


def scores(name: str, X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    if is_fomo(name):
        fn = fomo_scores
    else:
        fn = (kumo_scores if is_kumo(name) else tabpfn_scores) if is_gpu(name) else pyod_scores
    s = fn(name, X_train, X_test, seed)
    if s.shape != (len(X_test),) or not np.isfinite(s).all():
        raise ValueError(f"{name}: 점수 모양 {s.shape} 또는 NaN/inf {int((~np.isfinite(s)).sum())}개")
    return s
