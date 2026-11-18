"""C1 의 F2: TabArena Agent Claude Code × Sonnet +cont 풀이의 비율 피처 3열을 더한 16열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p07__claude-code-sonnet-cont__r1__20261116-161213 (r0f0 test AUC 0.7383, Is-this 의 Agent 최고) 의
작업 디렉터리. 최종 제출 = workspace/r.py: CatBoost (300번, depth 4) 를 seed 5개로 train 전체에 fit 해 평균. r2.py (LR 섞기)·
r3.py (피처 있고 없고 비교) 는 실험이고 제출을 안 씀. 피처는 r.py 의 f 그대로:
  ratio = credit_amount / income,  pm = credit_amount / credit_term,  mi = pm / income
빼 둔 것: CatBoost 모델·seed 평균. r2.py 의 log1p·one-hot (LR 입력, 최종 제출에 없음). 타깃 피처 없음.
train+test 로 fit 한 것: 없음 (행마다 계산).
동일성 확인 (로컬, 2026-11-18): 이 피처로 에이전트 r.py 의 CatBoost CV (StratifiedKFold(5, random_state=0), random_seed 0) 를
다시 돌려 기록 0.7605636995617349 와 비교 (docs/tabarena.md 의 "C1"). r3.py 의 기록으로는 이 3열의 CV 이득이 작음
(반복 5-fold 0.7445 대 원본만 0.7436). TFM 피처 상한 없음.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_is_this_f0")

TFM_MAX_FEATURES = None


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ratio"] = df.credit_amount / df.income
    out["pm"] = df.credit_amount / df.credit_term
    out["mi"] = out.pm / df.income
    return out.astype(float)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    return (pd.concat([X, ratios(train).reset_index(drop=True)], axis=1),
            pd.concat([X_test, ratios(test).reset_index(drop=True)], axis=1), cats)
