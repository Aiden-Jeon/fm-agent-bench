"""nomad2018 (MLE-bench) bandgap 타깃: formation 타깃과 같은 피처 (competitions/mleb-nomad2018-formation/features.py)."""

from __future__ import annotations

from fab.core.competition import load_competition, load_features

build = load_features(load_competition("mleb-nomad2018-formation")).build
