"""대회 공통 피처 유틸."""

from __future__ import annotations

import os
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


def _read_cache(cache: Path, ids: np.ndarray) -> dict[str, np.ndarray] | None:
    if not cache.exists():
        return None
    try:
        with np.load(cache) as z:
            if np.array_equal(z["_ids"], ids):
                return {k: z[k] for k in z.files if k != "_ids"}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:  # 쓰다 만 파일·없는 키 → 다시 계산
        print(f"[features] 캐시 읽기 실패 → 다시 계산: {e!r}")
    return None


def cached(name: str, ids: np.ndarray, data_dir: Path, compute: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """표 밖 피처처럼 무거운 계산을 data_dir/_features/<name>.npz 에 캐시 (행 id 순서가 같을 때만 씀).
    파일 잠금: 한 작업 안에서 모델마다 새 프로세스로 돌 때 동시에 시작해도 한 프로세스만 계산하고 나머지는 기다렸다가 캐시를 읽음.
    캐시 폴더·잠금을 못 쓰면 캐시 없이 계산, 캐시가 깨졌으면 다시 계산."""
    import fcntl

    cache = Path(data_dir) / "_features" / f"{name}.npz"
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        lock = open(cache.with_suffix(".lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
    except OSError as e:
        print(f"[features] 캐시 폴더·잠금을 못 씀 → 캐시 없이 계산: {e!r}")
        return compute()
    with lock:  # 닫으면 잠금도 풀림
        if (out := _read_cache(cache, ids)) is not None:
            return out
        out = compute()
        try:
            tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")  # 임시 파일 → 교체 (잠금 밖에서 읽는 경우 대비)
            np.savez(tmp, _ids=ids, **out)
            os.replace(tmp, cache)
        except OSError as e:  # 캐시는 있으면 좋은 것
            print(f"[features] 캐시 저장 실패 (무시): {e!r}")
    return out


def auto_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """문자열/불리언 컬럼은 category, 나머지는 float 로 변환. (df, cat_cols) 반환."""
    df = df.copy()
    cat_cols = [c for c in df.columns
                if df[c].dtype == object or df[c].dtype == bool or str(df[c].dtype) in ("string", "str", "category")]
    for c in cat_cols:
        df[c] = df[c].astype(str).where(df[c].notna()).astype("category")
    for c in df.columns.difference(cat_cols):
        df[c] = df[c].astype(float)
    return df, cat_cols


def split(df: pd.DataFrame):
    """build() 안에서 concat(keys=["train","test"]) 한 df 를 다시 나눔."""
    return df.loc["train"].reset_index(drop=True), df.loc["test"].reset_index(drop=True)
