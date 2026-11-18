"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 학업 비율 7열 + 범주 빈도 17열 (train 으로만 다시 fit) 을 더한 60열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p09__aide-sonnet__r1__20261116-182911 (r0f0 test logloss 0.5515, students 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 cbdaf801, 에이전트 CV 0.5210). 카드: competitions/ta-students-dropout-and-academic-success/c1.md
(PR #34). 최종 제출 = 8개 (CatBoost · LR · LightGBM · ExtraTrees · MLP · 수치 CatBoost (CBN) · LR 둘째 · 계층 LightGBM) 의 가중 산술·기하
평균·스태킹 평균 + 온도. 옮긴 것 (계산식은 에이전트 그대로):
  학업 비율  학기 (1st·2nd) 마다 승인 / (등록 + 0.001) · 승인 / (평가 + 1) 4열, 2학기 − 1학기 승인 비율, 승인 합, 성적 평균 = 7열
             (fe, best_solution.py:45-63)
  범주 빈도  범주 열 17개 (설정의 범주 = 에이전트 cat_cols: base_cat 10 + object 열 7, 에이전트 환경이 pandas 2.3.3 이라 글자 열이 object)
             의 값 빈도, <열>_f 17열 (Xn, best_solution.py:75-84, CBN 이 받음)
뺀 것: 범주 열의 번호 매김 (Xn 의 cat.codes, 모델 입력 인코딩), 8개 모델·섞기·스태킹 (OOF 예측 = 타깃)·온도. 타깃 피처 없음.
train+test fit: 에이전트는 범주 빈도를 train+test 로 셈 → 같은 계산을 train 행으로만 fit (PR #34 의 R3 해석, 2026-11-18 사용자).
test 에만 있는 값은 빈도 0.
동일성 확인 (로컬, 2026-11-18): 빈도는 에이전트 배열이 없어, fit 을 train+test 로 둔 판이 PR #35 판 (에이전트 식 그대로) 과 같은지 본 뒤
train 판과의 차이 (바뀐 값 수) 를 셈. 비율 7열은 에이전트 CatBoost 성분 CV (0.53378 대 기록 0.53332, PR #35) 그대로
(docs/tabarena.md 의 "C1"). TFM 피처 상한 없음 (60열).
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


def frequencies(train: pd.DataFrame, test: pd.DataFrame, cats: list[str], with_test: bool = False):
    """에이전트: 파일마다 astype(str) 한 값의 빈도. with_test=True 면 에이전트처럼 train+test 로 셈 (동일성 확인용)."""
    ftr, fte = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for c in cats:
        s_tr, s_te = train[c].astype(str), test[c].astype(str)
        vc = pd.concat([s_tr, s_te]).value_counts() if with_test else s_tr.value_counts()
        ftr[c + "_f"] = s_tr.map(vc).fillna(0).astype(float)
        fte[c + "_f"] = s_te.map(vc).fillna(0).astype(float)
    return ftr.reset_index(drop=True), fte.reset_index(drop=True)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    rat_tr, rat_te = split(ratios(pd.concat([train, test], keys=["train", "test"])))
    f_tr, f_te = frequencies(train, test, cats)
    return pd.concat([X, rat_tr, f_tr], axis=1), pd.concat([X_test, rat_te, f_te], axis=1), cats
