"""알아볼 수 없게 바꾼 대회 (docs/tabarena.md 의 "범주 대 오염"): 원래 데이터셋과 같은 OpenML 데이터·split·행을
알아볼 수 없게 바꿈. 설정의 tabarena.obfuscate ({seed, hide}) 가 있는 대회만.

- 열 이름: 피처는 원래 순서대로 f0, f1, … (순서는 그대로), 타깃은 target
- 범주 값: 열마다 따로, 무작위 토큰 (영소문자로 시작하는 5글자 영소문자·숫자). 같은 열이면 train·test·모든 split 에서 같은 대응
  (데이터 전체의 값으로 만듦). 원래 값의 순서와 토큰 순서는 무관 (값 → 토큰 배정도 무작위). 결측은 결측 (pandas 가 결측으로
  읽는 글자 'nan' 등도 결측: 원래 public 을 pandas 로 읽은 것과 같은 자리)
- 수치 열: 이름만 바꾸고 값은 그대로
- 클래스: c0, c1, … (원래 클래스 이름의 정렬 순서대로 → 이진의 양성 클래스·다중의 열 순서가 원래와 같음)
- 행 id: 원래와 같은 방식 (데이터셋·split 의 고정 seed 순열) 이지만 seed 에 obfuscate.seed 를 더해 원래 대회와 다른 id

대응표는 FAB_TABARENA/data/obfuscate/<대회>.json 에만 (정답처럼 에이전트에게 안 보임). seed 와 이 코드로 다시 만들 수 있음.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES

SUFFIX = "~obf"  # 결과표의 dataset 키 (원래 데이터셋 이름 + SUFFIX) · CLI 의 대상 이름
TARGET = "target"
TOKEN = re.compile(r"[a-z][a-z0-9]{4}")
_ALPHA, _ALNUM = "abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz0123456789"
# pandas 가 결측·불리언으로 읽는 글자 (5글자 영소문자·숫자에는 'false' 뿐이지만 같이 막음)
_BAD = {s.lower() for s in STR_NA_VALUES} | {"true", "false"}


def is_obf(key: str) -> bool:
    return key.endswith(SUFFIX)


def base(key: str) -> str:
    return key.removesuffix(SUFFIX)


def result_key(ta: dict) -> str:
    """결과표 (tabarena.csv) 의 dataset 열: 바꾼 대회는 원래 이름 + SUFFIX → compare·elo 의 원래 데이터셋과 안 섞임."""
    return ta["dataset"] + (SUFFIX if ta.get("obfuscate") else "")


def feature_names(features: list[str]) -> dict[str, str]:
    return {c: f"f{i}" for i, c in enumerate(features)}


def class_names(classes: list[str]) -> dict[str, str]:
    """원래 대회의 클래스 순서 (prepare 가 CSV 를 pandas 로 읽은 값의 정렬: 모두 숫자면 숫자로, 아니면 글자로) 대로
    c0, c1, … (10개 이상이면 0 을 채워 글자 정렬 = 숫자 순서) → 이진의 양성 클래스·다중의 열 순서가 원래와 같음."""
    num = pd.to_numeric(pd.Series(classes, dtype=object), errors="coerce")
    order = [c for _, c in sorted(zip(num, classes))] if num.notna().all() else sorted(classes)
    w = len(str(len(classes) - 1))
    return {c: f"c{i:0{w}d}" for i, c in enumerate(order)}


def missing(s: pd.Series) -> pd.Series:
    """결측 자리: 빈칸 + pandas 기본값이 결측으로 읽는 글자 (TabArena 데이터의 범주 'nan' 은 원래 결측, docs/tabarena.md 9)."""
    return s.isna() | s.astype(str).isin(STR_NA_VALUES)


def tokens(rng: np.random.Generator, n: int, avoid: set[str]) -> list[str]:
    out, seen = [], set(avoid)
    while len(out) < n:
        t = _ALPHA[rng.integers(26)] + "".join(_ALNUM[i] for i in rng.integers(36, size=4))
        if t not in seen and t not in _BAD:
            seen.add(t)
            out.append(t)
    return out


def build_map(df: pd.DataFrame, features: list[str], target: str, cats: list[str], task_id: int, seed: int,
              classification: bool) -> dict:
    """대응표: {columns: {원래: 새}, target: [원래, 새], classes: {원래: 새}, values: {새 열: {원래 값: 토큰}}}.
    열마다 rng = (task_id, seed, 열 위치). 원래 값 (문자열) 을 정렬한 뒤 무작위 순서로 토큰을 배정. 토큰은 그 열의 원래 값과
    겹치지 않게 (원래 값이 우연히 토큰 꼴이어도)."""
    cols = feature_names(features)
    values = {}
    for i, c in enumerate(features):
        if c not in cats:
            continue
        rng = np.random.default_rng([task_id, seed, i])
        s = df[c]
        vals = sorted(set(s[~missing(s)].astype(str)))
        toks = tokens(rng, len(vals), set(vals))
        values[cols[c]] = dict(zip([vals[j] for j in rng.permutation(len(vals))], toks))
    y = df[target]
    classes = class_names(sorted(set(y[~missing(y)].astype(str)))) if classification else None
    return dict(columns=cols, target=[target, TARGET], classes=classes, values=values)


def apply(df: pd.DataFrame, m: dict) -> pd.DataFrame:
    """원래 표 (read_data 의 열: 피처 + 타깃) → 바꾼 표. 범주 열은 대응표로 (결측은 결측), 수치 열은 값 그대로."""
    out = df.rename(columns={**m["columns"], m["target"][0]: m["target"][1]})
    for c, mp in m["values"].items():
        s = out[c]
        out[c] = s.astype(str).map(mp).where(~missing(s))
        if (bad := out[c].isna() & ~missing(s)).any():
            raise SystemExit(f"{c}: 대응표에 없는 값 {sorted(set(s[bad].astype(str)))[:5]}")
    if m["classes"]:
        y = out[TARGET]
        out[TARGET] = y.astype(str).map(m["classes"]).where(~missing(y))
    return out


def leak_terms(comp, m: dict) -> tuple[list[str], list[str]]:
    """바꾼 대회의 설명·열 머리에 있으면 안 되는 글: (대소문자 무시: 원래 열 이름 (3글자 이상)·타깃 이름·설정의 hide 말,
    대소문자 그대로: 원래 클래스 이름)."""
    ob = comp.tabarena["obfuscate"]
    loose = [c for c in m["columns"] if len(c) >= 3] + [m["target"][0], *ob.get("hide", [])]
    return sorted(set(loose)), sorted(m["classes"] or [])


def find_leaks(comp, m: dict, texts: dict[str, str]) -> list[str]:
    loose, exact = leak_terms(comp, m)
    found = []
    for where, text in texts.items():
        for t, flags in [*((t, re.I) for t in loose), *((t, 0) for t in exact)]:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", text, flags):
                found.append(f"{where}: '{t}'")
    return found
