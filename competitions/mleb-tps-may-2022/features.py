"""tabular-playground-series-may-2022 (MLE-bench): 원본 열만 (F0).

f_27 (10글자 문자열, train 80만 행에 고유값 약 67만 개) 는 그대로 범주형 하나로 둠 → TFM 에는 정수 코드(사전순).
글자 위치별로 나누는 건 사람 피처 ~f1 (features/f1.py). xgb 는 이 열 때문에 느림 (로컬 8코어 64만 행 20트리 28초).
"""

from __future__ import annotations

import pandas as pd

from fab.core.features import auto_types, split


def build(train: pd.DataFrame, test: pd.DataFrame):
    df, cats = auto_types(pd.concat([train.drop(columns="target"), test], keys=["train", "test"]).drop(columns="id"))
    return *split(df), cats
