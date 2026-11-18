"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 학업 비율 7열 + 범주 빈도 17열을 더한 60열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p09__aide-sonnet__r1__20261116-182911 (r0f0 test logloss 0.5515, students 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 cbdaf801, 에이전트 CV 0.5210). 최종 제출 = 8개 (CatBoost · LR · LightGBM · ExtraTrees ·
MLP · 수치 CatBoost (CBN) · LR 둘째 · 계층 LightGBM) 의 가중 산술·기하 평균·스태킹 평균 + 온도. 피처는 fe 와 CBN·MLP 입력 Xn:
  학업 비율  학기 (1st·2nd) 마다 승인 / (등록 + 0.001) · 승인 / (평가 + 1) 4열, 2학기 − 1학기 승인 비율, 승인 합, 성적 평균 = 7열 (fe)
  범주 빈도  범주 열 17개 (설정의 범주 = 에이전트 cat_cols) 의 값 빈도 (train+test 전체), <열>_f 17열 (Xn, CBN·MLP 가 씀)
빼 둔 것: 범주 열의 번호 매김 (Xn 의 cat.codes, 모델 입력 인코딩), 8개 모델·섞기·스태킹 (OOF 예측 = 타깃)·온도. 타깃 피처 없음.
train+test 로 fit 한 것: 범주 빈도 (값 빈도를 train+test 로 셈, 에이전트 그대로).
동일성 확인 (로컬, 2026-11-18): 이 피처로 에이전트 CatBoost 성분 (StratifiedKFold(5, random_state=42), seed 42+fold, 검증 fold 로
early stopping) 을 다시 돌려 기록 CV 0.5333204674992351 과 비교 (docs/tabarena.md 의 "C1"). TFM 피처 상한 없음 (60열).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fab.core.competition import load_module
from fab.core.features import split

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_students_f0")

TFM_MAX_FEATURES = None


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for s in ["1st", "2nd"]:
        e = df[f"Curricular_units_{s}_sem_enrolled"]
        a = df[f"Curricular_units_{s}_sem_approved"]
        ev = df[f"Curricular_units_{s}_sem_evaluations"]
        out[f"appr_rate_{s}"] = a / (e + 1e-3)
        out[f"appr_eval_{s}"] = a / (ev + 1)
    out["appr_diff"] = out["appr_rate_2nd"] - out["appr_rate_1st"]
    out["appr_total"] = df["Curricular_units_1st_sem_approved"] + df["Curricular_units_2nd_sem_approved"]
    out["grade_mean"] = (df["Curricular_units_1st_sem_grade"] + df["Curricular_units_2nd_sem_grade"]) / 2
    return out.astype(float)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    raw = pd.concat([train, test], keys=["train", "test"])
    new = ratios(raw)
    for c in cats:  # 에이전트: astype(str) 값의 train+test 빈도 (F0 의 범주 값도 같은 문자열 분할)
        s = raw[c].astype(str)
        new[c + "_f"] = s.map(s.value_counts()).astype(float)
    new_tr, new_te = split(new)
    return pd.concat([X, new_tr], axis=1), pd.concat([X_test, new_te], axis=1), cats
