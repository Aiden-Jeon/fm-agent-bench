"""C1 의 F2 (작게): 원본 열 + 격자 파생 + 대칭 + RDF·이웃 통계, 약 1.1천 열. 에이전트(Codex × GPT) 피처 중 사람이 읽을 수 있는
물리 피처만 (_geo.py). TFM 피처 상한 없음: 대회 설정의 tfm_max_features 를 쓰지 않고 TFM 에 열을 다 넣음
(GBDT 중요도로 고르면 "TFM 에 도움" 이 GBDT 가 고른 덕일 수 있어서)."""

from __future__ import annotations

from pathlib import Path

from fab.core.competition import load_module

_geo = load_module(Path(__file__).with_name("_geo.py"), "features_nomad_geo")

PIP = _geo.PIP
TFM_MAX_FEATURES = None
BLOCKS = ["symmetry", "radial"]


def build(train, test, data_dir):
    return _geo.build(train, test, data_dir, BLOCKS)
