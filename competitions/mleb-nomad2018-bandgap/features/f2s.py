"""nomad2018 bandgap 타깃: formation 과 같은 피처 변형 (competitions/mleb-nomad2018-formation/features/f2s.py)."""

from __future__ import annotations

from fab.core.competition import load_competition, load_features

_mod = load_features(load_competition("mleb-nomad2018-formation"), "f2s")
PIP, TFM_MAX_FEATURES, BLOCKS, build = _mod.PIP, _mod.TFM_MAX_FEATURES, _mod.BLOCKS, _mod.build
