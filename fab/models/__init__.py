"""모델 레지스트리. 모든 모델은 sklearn 스타일 fit / predict(_proba) 를 가짐.

새 모델 추가: 아래 REGISTRY 에 `이름: (모듈, 팩토리 함수명)` 등록, 원격 설치가 필요하면 REMOTE_PIP 에도 추가.
팩토리 시그니처: make(task: str, cat_cols: list[str], seed: int, **설정) -> estimator (설정은 VARIANTS 에서 전달)

이름 뒤에 `@<spec>` 을 붙이면 컨텍스트 선택 래퍼로 감쌈 (예: tabpfn_v2@knn1000, models/context.py 참고).
맨 뒤에 `~<variant>` 를 붙이면 그 실행만 피처 변형 competitions/<대회>/features/<variant>.py 를 씀 (예: catboost~f2,
tabpfn_v2@knn1000~f2). 실행 이름(결과 폴더·results.csv·채점의 model)은 `~` 까지 포함한 전체 이름.
"""

from __future__ import annotations

import importlib
import inspect

REGISTRY = {
    "lgbm": ("fab.models.gbdt", "make_lgbm"),
    "xgb": ("fab.models.gbdt", "make_xgb"),
    "catboost": ("fab.models.gbdt", "make_catboost"),
    "tabpfn": ("fab.models.tfm", "make_tabpfn"),  # 최신 기본 버전 (첫 실행 시 브라우저 라이선스 동의 필요)
    "tabpfn_v2": ("fab.models.tfm", "make_tabpfn_v2"),  # 라이선스 동의 없이 사용 가능
    "tabpfn35": ("fab.models.tfm", "make_tabpfn35"),  # TabPFN-3.5 명시 (라이선스 동의 필요)
    "tabicl": ("fab.models.tfm", "make_tabicl"),
    "limix2m": ("fab.models.tfm", "make_limix2m"),  # 2M 경량판, 비상업 라이선스
    "mitra": ("fab.models.tfm", "make_mitra"),  # Mitra-v2 zero-shot
    "mitra_ft": ("fab.models.tfm", "make_mitra_ft"),  # Mitra-v2 + 50 step fine-tuning
    "exaone": ("fab.models.tfm", "make_exaone"),  # 가중치 비상업 라이선스
    "kumo": ("fab.models.tfm", "make_kumo"),  # NVIDIA Kumo-Tabular (large)
    "causilo": ("fab.models.tfm", "make_causilo"),  # Nums AI Causilo, 가중치 비상업 라이선스
    "sap_rpt": ("fab.models.tfm", "make_sap_rpt"),  # SAP-RPT-1-OSS 컨텍스트 2048 (gated, 연구용 라이선스)
}

_LIMIX = [["--no-deps", "git+https://github.com/limix-ldm-ai/LimiX.git"], ["einops", "kditransform", "nvtx"],
          ["-U", "numba"]]  # 기본 numba 가 오래된 환경 는 numpy<2.3 만 지원 → 다른 패키지가 numpy 를 올리면 kditransform 이 깨짐

# 원격 실행(fab databricks)에서 모델별로 추가 설치할 pip 인자 목록 (항목마다 pip install 한 번)
REMOTE_PIP = {
    "tabpfn": [["tabpfn"]],
    "tabpfn_v2": [["tabpfn"]],
    "tabpfn35": [["tabpfn"]],
    "tabicl": [["tabicl"]],
    "limix2m": _LIMIX,  # torch==2.9.1 고정을 피하려고 --no-deps 로 설치
    "mitra": [["autogluon.tabular[mitra]>=1.6"]],
    "mitra_ft": [["autogluon.tabular[mitra]>=1.6"]],
    "exaone": [["exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"]],
    "kumo": [["structured-data-models"]],
    "causilo": [["causilo"]],
    "sap_rpt": [["git+https://github.com/SAP-samples/sap-rpt-1-oss"]],
}

# HF gated 가중치 (repo, 파일). 원격에는 HF 토큰을 보내지 않으므로 `fab databricks` 가 로컬에서 받아 Volume 에 올림
GATED = {"sap_rpt": [("SAP/sap-rpt-1-oss", "2025-11-04_sap-rpt-one-oss.pt")]}


# 설정만 다른 변형: 이름 → (REGISTRY 모델, 인자). kumo_e4 는 앙상블 크기 벤치마크용 (기본값을 정하면 정리)
# 인자 중 max_features 는 대회 설정 tfm_max_features 대신 쓰고, 나머지는 팩토리로 전달
VARIANTS = {
    "kumo_e4": ("kumo", dict(n_estimators=4)),
    "sap_rpt8k": ("sap_rpt", dict(max_context_size=8192)),  # README 기본값, 80GB GPU 용
}


FEATURES_SEP = "~"


def split_features(name: str) -> tuple[str, str | None]:
    """실행 이름 → (모델 이름, 피처 변형). 변형이 없으면 None (대회의 기본 features.py)."""
    model, sep, variant = name.partition(FEATURES_SEP)
    if sep and not variant:
        raise ValueError(f"'{name}': {FEATURES_SEP} 뒤에 피처 변형 이름이 없음")
    return model, variant or None


def base_name(name: str) -> str:
    base = split_features(name)[0].partition("@")[0]
    return VARIANTS.get(base, (base,))[0]


def make_model(name: str, task: str, cat_cols: list[str], seed: int, max_rows: int | None = None,
               max_features: int | None = None, metric: str | None = None):
    """max_rows / max_features: TFM 컨텍스트 행·피처 상한 (대회 설정 tfm_max_rows / tfm_max_features).
    GBDT 와 `@spec` 래퍼(컨텍스트를 직접 고름)에는 적용 안 함.
    metric: 대회 지표 (팩토리가 metric 인자를 받을 때만 전달. 예: kumo 는 mae 면 분포의 중앙값으로 예측)."""
    base, _, spec = split_features(name)[0].partition("@")  # 피처 변형은 fab run 이 피처를 만들 때 씀
    base, kwargs = VARIANTS.get(base, (base, {}))
    kwargs = dict(kwargs)
    max_features = kwargs.pop("max_features", max_features)
    if base not in REGISTRY:
        raise KeyError(f"알 수 없는 모델 '{base}'. 사용 가능: {', '.join([*REGISTRY, *VARIANTS])}")
    module, fn = REGISTRY[base]
    factory = getattr(importlib.import_module(module), fn)
    params = inspect.signature(factory).parameters
    if "metric" in params:
        kwargs["metric"] = metric
    if not spec:
        from fab.models.tfm import _TFM

        model = factory(task, cat_cols, seed, **kwargs)
        if isinstance(model, _TFM):
            model.set_params(max_rows=max_rows, max_features=max_features, task=task)
        return model
    from fab.models.context import wrap

    return wrap(lambda: factory(task, cat_cols, seed, **kwargs), spec, task, cat_cols, seed)
