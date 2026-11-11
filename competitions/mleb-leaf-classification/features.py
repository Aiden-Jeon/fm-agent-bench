"""leaf-classification (MLE-bench): 원본 열만 (F0). 잎 이미지(images/*.jpg)는 안 씀.

margin·shape·texture 각 64개 = 192개, 모두 수치형 (대회가 이미지에서 미리 뽑아 둔 기술자).
"""

from __future__ import annotations

import pandas as pd

from fab.core.features import split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train.drop(columns="species"), test], keys=["train", "test"]).drop(columns="id").astype(float)
    return *split(df), []
