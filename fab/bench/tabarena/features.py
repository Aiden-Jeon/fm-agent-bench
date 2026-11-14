"""TabArena split 대회의 기본 피처 (F0): 원본 열 그대로, 범주형만 표시.

대회 폴더의 features.py (FEATURES_PY, split_comp 가 생성 폴더에도 같은 글로 씀) 가 build_for 를 부름.
범주형 = 설정의 tabarena.categorical (OpenML 범주형 표시 + 문자열 열). Amazon 의 id 처럼 숫자로 보이는 범주도 범주로
(TabArena 와 같은 표시). 나머지는 float.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from fab.core.features import split

FEATURES_PY = '''\
"""TabArena split 대회의 기본 피처 (F0, fab/bench/tabarena/features.py)."""

from pathlib import Path

from fab.bench.tabarena.features import build_for


def build(train, test):
    return build_for(Path(__file__).with_name("config.yaml"), train, test)
'''


def as_category(s: pd.Series) -> pd.Series:
    """범주 값을 문자열로. train·test 를 따로 읽어 같은 열이 한쪽은 int, 한쪽은 float (결측 때문에) 일 수 있음 → 정수 값은
    '1.0' 이 아니라 '1' 로 (두 파일의 범주가 같은 값이 되게)."""
    if pd.api.types.is_numeric_dtype(s):
        s = s.map(lambda v: v if pd.isna(v) else str(int(v)) if float(v).is_integer() else repr(float(v)))
    else:
        s = s.astype(str).where(s.notna())
    return s.astype("category")


def build_for(config: Path, train: pd.DataFrame, test: pd.DataFrame):
    cfg = yaml.safe_load(Path(config).read_text())
    cats = set(cfg["tabarena"]["categorical"])
    df = pd.concat([train.drop(columns=cfg["target"]), test], keys=["train", "test"]).drop(columns=cfg["id_col"])
    for c in df.columns:
        df[c] = as_category(df[c]) if c in cats else df[c].astype(float)
    return *split(df), [c for c in df.columns if c in cats]
