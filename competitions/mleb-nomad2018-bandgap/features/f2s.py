"""nomad2018 bandgap 타깃: formation 과 같은 피처 변형 (competitions/mleb-nomad2018-formation/features/f2s.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "features_mleb_nomad2018_formation_f2s", Path(__file__).parents[2] / "mleb-nomad2018-formation" / "features" / "f2s.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PIP, TFM_MAX_FEATURES, BLOCKS, build = _mod.PIP, _mod.TFM_MAX_FEATURES, _mod.BLOCKS, _mod.build
