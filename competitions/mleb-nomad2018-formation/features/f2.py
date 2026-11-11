"""C1 의 F2 (전부): f2s + SOAP 평균·표준편차·위상·푸리에·스펙트럼, 약 3.9만 열. 에이전트(Codex × GPT)의 타깃 비의존 피처 전부
(_geo.py). TFM 에는 fold 학습 데이터의 LightGBM 중요도 상위 500열만 (TabPFN v2 사전 학습 범위, bioresponse 와 같은 값)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("features_nomad_geo", Path(__file__).with_name("_geo.py"))
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)

PIP = _geo.PIP
TFM_MAX_FEATURES = 500
BLOCKS = ["symmetry", "radial", "soap_mean", "soap_std", "topology", "fourier", "spectrum"]


def build(train, test, data_dir):
    return _geo.build(train, test, data_dir, BLOCKS)
