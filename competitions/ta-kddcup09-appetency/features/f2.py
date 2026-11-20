"""C1 의 F2: TabArena Agent AIDE × Sonnet 풀이의 범주 값 빈도 38열 (train 으로만 다시 fit) 을 더한 250열 (docs/tabarena.md 의 "행 추가: kddcup09_appetency").

출처: 실행 tabarena-p16__aide-sonnet__r1__20261119-215413 (r0f0 test 1 − AUC 0.1559, kddcup09 의 Agent 최고) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 24a44778, 에이전트 OOF 블렌드 AUC 0.8477). 카드: competitions/ta-kddcup09-appetency/c1.md.
최종 제출 = LGB 0.25 · XGB 0.25 · CatBoost 4개 0.5 의 OOF 순위 평균. 옮긴 것 (계산식은 에이전트 그대로):
  범주 빈도  범주 열 38개 (설정의 범주 = 에이전트의 object 열) 마다 결측을 "NA" 로 채운 값의 개수, <열>_freq 38열
             (best_solution.py:18-20, 세 성분 모두 받음)
뺀 것: 범주 열의 타깃 인코딩 38열 (best_solution.py:38-56, 타깃 의존), 그 키인 범주 코드 (모델 입력 아님), 모델·seed 평균·순위 섞기 무게.
train+test fit: 에이전트는 빈도를 train+test 로 셈 → 같은 계산을 train 행으로만 fit (PR #34 의 R3 해석, 2026-11-18 사용자).
test 에만 있는 값은 빈도 0, 결측은 train 의 결측 수.
동일성 확인 (로컬): with_test=True 판으로 에이전트 CatBoost 성분 (best_solution.py:130-147 의 설정 · fold · seed) 을 우리 코드로
다시 돌려 기록 OOF AUC 0.8440615983048002 와 견준 뒤 train 판으로 바꿈 (docs/tabarena.md). TFM 피처 상한 없음 (250열).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_kddcup09_f0")

TFM_MAX_FEATURES = None


def frequencies(train: pd.DataFrame, test: pd.DataFrame, cats: list[str], with_test: bool = False):
    """에이전트: allx[c].fillna("NA").value_counts() 를 map. with_test=True 면 에이전트처럼 train+test 로 셈 (동일성 확인용)."""
    ftr, fte = pd.DataFrame(index=range(len(train))), pd.DataFrame(index=range(len(test)))
    for c in cats:
        s_tr, s_te = train[c].fillna("NA").astype(str), test[c].fillna("NA").astype(str)
        vc = pd.concat([s_tr, s_te]).value_counts() if with_test else s_tr.value_counts()
        ftr[c + "_freq"] = s_tr.map(vc).fillna(0).astype(float).to_numpy()
        fte[c + "_freq"] = s_te.map(vc).fillna(0).astype(float).to_numpy()
    return ftr, fte


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    f_tr, f_te = frequencies(train, test, cats)
    return pd.concat([X, f_tr], axis=1), pd.concat([X_test, f_te], axis=1), cats
