"""tps-may-2022 사람 피처 (F1): F0 에서 f_27 만 글자 위치별 10열 (A=0, B=1, ...) + 서로 다른 글자 수로 바꿈.

대회 당시 공개 노트북에서 널리 쓴 처리 (타깃 비의존). 나머지 열은 F0 (features.py) 의 출력을 그대로 씀.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_tps_may_2022_f0")
N_CHARS = 10


def chars(s: pd.Series) -> np.ndarray:
    """f_27 (길이 10 ASCII 대문자) → (n, 10) 글자 코드 (A=0). 한 번에 바이트로 풀어 reshape."""
    s = s.astype(str)
    if not (s.str.len() == N_CHARS).all():
        raise ValueError(f"f_27 길이가 {N_CHARS} 이 아닌 행이 있음")
    return (np.frombuffer("".join(s).encode("ascii"), np.uint8).reshape(-1, N_CHARS) - ord("A")).astype(np.int16)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, _ = _f0.build(train, test)
    out = []
    for df in (X, X_test):
        c = chars(df.pop("f_27"))
        c_sorted = np.sort(c, axis=1)
        nunique = 1 + (np.diff(c_sorted, axis=1) != 0).sum(1)
        new = pd.DataFrame(c, columns=[f"f_27_ch{i}" for i in range(N_CHARS)], index=df.index).assign(f_27_nunique=nunique)
        out.append(pd.concat([df, new.astype(float)], axis=1))
    return *out, []
