"""tps-may-2022 사람 피처 (F1): f_27 을 글자 위치별 10열 (A=0, B=1, ...) + 서로 다른 글자 수로 바꾸고 원래 열은 뺌.

대회 당시 공개 노트북에서 널리 쓴 처리 (타깃 비의존). 나머지 원본 열은 F0 와 같음.
"""

from __future__ import annotations

import pandas as pd

from fab.core.features import split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df = pd.concat([train.drop(columns="target"), test], keys=["train", "test"]).drop(columns="id")
    s = df.pop("f_27")
    for i in range(10):
        df[f"f_27_ch{i}"] = s.str[i].map(ord) - ord("A")
    df["f_27_nunique"] = s.map(lambda v: len(set(v)))
    return *split(df.astype(float)), []
