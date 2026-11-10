"""nomad2018 (MLE-bench) bandgap 타깃: formation 타깃과 같은 피처 (competitions/mleb-nomad2018-formation/features.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "features_mleb_nomad2018_formation", Path(__file__).parents[1] / "mleb-nomad2018-formation" / "features.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build = _mod.build
