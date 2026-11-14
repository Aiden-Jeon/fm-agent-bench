"""TabArena split 대회의 기본 피처 (F0, fab/bench/tabarena/features.py)."""

from pathlib import Path

from fab.bench.tabarena.features import build_for


def build(train, test):
    return build_for(Path(__file__).with_name("config.yaml"), train, test)
