"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 파생 23열 (빈도는 train 으로만 다시 fit) 을 더한 70열 (docs/tabarena.md 의 "행 추가: Diabetes130US").

출처: 실행 tabarena-p19__aide-sonnet__r1__20261123-113000 (r0f0 test 1 − AUC 0.31615, Diabetes130US 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 a0288e4d, step 11, 에이전트 OOF 블렌드 AUC 0.66801). 카드: competitions/ta-diabetes130us/c1.md.
최종 제출 = CatBoost 3설정의 순위 평균과 LightGBM 을 OOF 로 고른 무게로 섞음. 옮긴 것 (계산식은 에이전트 그대로, best_solution.py:41-71):
  행 단위 9열   total_prior · prior_ei (이전 방문 합) · n_meds_active · n_meds_changed · n_up · n_down (약 23열의 상태 개수) ·
                age_num (나이 구간의 가운데) · lab_per_day · med_per_day (입원 일수로 나눔)
  진단 12열     diag_1..3 마다 _cat (코드 앞 3글자, 범주) · _num (앞의 V/E 를 뗀 수치) · _freq (코드 빈도) · _cat_freq (앞 3글자 빈도)
  조합 2열      diag_combo (diag_1_cat + diag_2_cat, 범주) · spec_disp (medical_specialty + discharge_disposition_id, 범주)
뺀 것: 모델 · seed · 순위 섞기 무게 (OOF 정답으로 고름). 타깃 의존 블록은 없음. 이 실행의 다른 노드에 있던 ICD-9 장 묶음
(785~787 을 넣은 Strack 식) 은 최종 노드에 없어 옮기지 않음.
train+test fit: 에이전트는 빈도 4종을 train+test 로 셈 → 같은 계산을 train 행으로만 fit (R3 해석, 2026-11-18 사용자).
test 에만 있는 값은 빈도 0. 결측 진단 코드는 에이전트처럼 "NA" 한 값으로 셈.
동일성 확인 (로컬): with_test=True 판으로 에이전트 LightGBM 성분 (best_solution.py:121-146 의 설정 · fold · seed) 을 우리 코드로
다시 돌려 기록 CV AUC 와 견준 뒤 train 판으로 바꿈 (docs/tabarena.md). TFM 피처 상한 없음 (70열).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_diabetes130us_f0")

TFM_MAX_FEATURES = None
MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
        "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"]
DIAGS = ["diag_1", "diag_2", "diag_3"]
NEW_CATS = ["diag_1_cat", "diag_2_cat", "diag_3_cat", "diag_combo", "spec_disp"]


def row_features(df: pd.DataFrame) -> pd.DataFrame:
    """행마다 계산 (fit 없음). df = pandas 기본값으로 읽은 train/test (에이전트와 같은 읽기)."""
    f = pd.DataFrame(index=df.index)
    f["total_prior"] = df.number_outpatient + df.number_emergency + df.number_inpatient
    f["prior_ei"] = df.number_emergency + df.number_inpatient
    meds = [m for m in MEDS if m in df.columns]
    f["n_meds_active"] = sum((df[m].astype(str) != "No").astype(int) for m in meds)
    f["n_meds_changed"] = sum(df[m].astype(str).isin(["Up", "Down"]).astype(int) for m in meds)
    f["n_up"] = sum((df[m].astype(str) == "Up").astype(int) for m in meds)
    f["n_down"] = sum((df[m].astype(str) == "Down").astype(int) for m in meds)
    f["age_num"] = df["age"].astype(str).str.extract(r"\[(\d+)")[0].astype(float) + 5
    f["lab_per_day"] = df.num_lab_procedures / df.time_in_hospital
    f["med_per_day"] = df.num_medications / df.time_in_hospital
    for d in DIAGS:
        s = df[d].fillna("NA").astype(str)
        f[d + "_cat"] = s.str[:3]
        f[d + "_num"] = pd.to_numeric(s.str.replace(r"^[VE]", "", regex=True), errors="coerce")
    f["diag_combo"] = f.diag_1_cat + "_" + f.diag_2_cat
    # 에이전트 (pandas 2.3) 는 결측 discharge 를 astype(str) 로 "nan" 글자로 붙임. pandas 3 은 결측으로 남겨 조합이 통째로
    # 결측이 되므로 "nan" 을 직접 넣음 (동일성 확인에서 찾음, train 1,651행)
    f["spec_disp"] = (df["medical_specialty"].fillna("NA").astype(str) + "_"
                      + df["discharge_disposition_id"].fillna("nan").astype(str))
    return f


def blocks(train: pd.DataFrame, test: pd.DataFrame, with_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """덧붙일 23열 (열 순서는 에이전트와 같음). with_test=True 면 에이전트처럼 빈도를 train+test 로 셈 (동일성 확인용)."""
    a, b = row_features(train), row_features(test)
    for d in DIAGS:
        for src, name in ((d, d + "_freq"), (d + "_cat", d + "_cat_freq")):
            s_tr = train[d].fillna("NA").astype(str) if src == d else a[src]
            s_te = test[d].fillna("NA").astype(str) if src == d else b[src]
            vc = pd.concat([s_tr, s_te]).value_counts() if with_test else s_tr.value_counts()
            a[name] = s_tr.map(vc).fillna(0).astype(float).to_numpy()
            b[name] = s_te.map(vc).fillna(0).astype(float).to_numpy()
    order = (["total_prior", "prior_ei", "n_meds_active", "n_meds_changed", "n_up", "n_down", "age_num", "lab_per_day",
              "med_per_day"] + [f"{d}{s}" for d in DIAGS for s in ("_cat", "_num", "_freq", "_cat_freq")]
             + ["diag_combo", "spec_disp"])
    return a[order], b[order]


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    a, b = blocks(train, test)
    for c in NEW_CATS:  # 범주 5열: F0 과 같은 category dtype (train+test 값의 합집합, 값 집합만 맞춤)
        levels = sorted(set(a[c]) | set(b[c]))
        a[c] = pd.Categorical(a[c], categories=levels)
        b[c] = pd.Categorical(b[c], categories=levels)
    a.index, b.index = X.index, X_test.index
    return pd.concat([X, a], axis=1), pd.concat([X_test, b], axis=1), [*cats, *NEW_CATS]

