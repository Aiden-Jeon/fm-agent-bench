"""nomad2018 (MLE-bench): 원본 열만 (F0). 결정 구조 파일(geometry.xyz)은 아직 안 씀.

두 타깃 대회(mleb-nomad2018-formation / -bandgap)가 같은 피처를 씀.
"""

from __future__ import annotations

import pandas as pd

from fab.core.features import split

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train.drop(columns=TARGETS), test], keys=["train", "test"]).drop(columns="id").astype(float)
    df["spacegroup"] = df["spacegroup"].astype(int).astype(str).astype("category")
    return *split(df), ["spacegroup"]
