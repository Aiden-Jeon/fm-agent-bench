"""C1 의 F2: TabArena Agent Codex × GPT 풀이의 트리 성분 입력 (타깃 비의존), 1,165열 (docs/tabarena.md 의 "C1").

출처: 실행 tabarena-p08__codex-gpt__r1__20261116-172003 (r0f0 test logloss 0.06179, splice 의 Agent 최고) 의 작업 디렉터리.
최종 제출 (submit_final.py) = 성분 14개 섞기 (가중치 ×0.8): 생성 모델 9종 (gen·pair·word·wordgen·wordgen2+codon·joint·gate·
wordprior, 위치별 k-mer 의 Markov·나이브 베이즈 꼴) + SVM (스펙트럼 커널) + 신경망 + 트리 3개 (cat 0.05·xgb 0.05·ext 0.1) + LR 0.02,
그 위에 최근접 train 행 투표 보정·1.2제곱·평탄화. 피처를 받는 표형 성분은 트리 셋 (+ LR) 뿐 → 트리 성분 입력의 합집합을 옮김.
계산식·순서는 에이전트 그대로 (같은 정의는 한 번만):
  원본      60열 (범주, F0). 에이전트는 one-hot (287 수준) 으로 넣음
  ext       extended_trees.py 의 수치 1,023열 (로그 "features (3190, 1310)" − one-hot 287):
            창 (길이 5·10·15·20·30, 시작 5칸씩) × 염기 묶음 (A·C·G·T·CT·AG·CG·AT) 개수 392, 창 6개 × 묶음 4개의 최장 연속 24,
            창 6개의 2–3-gram 개수 607 (CountVectorizer, 어휘는 train+test)
  cat       repeated.py 의 Xbase 중 ext 에 없는 창 (15,28)·(0,60) 의 A·C·G·T 개수 8열 (나머지 창 9개는 ext 와 같은 열)
  xgb       repeated.py 의 Xmotif: 위치 20–38 의 2-mer 19 · 위치 20–37 의 3-mer 18 · 위치 25–34 의 두 칸 이상 떨어진 쌍 36 = 73열
            (범주. 에이전트는 one-hot 으로 넣음)
  motif     위치 28·29 가 AG 인지 + 2 × 위치 30·31 이 GT 인지 (0–3), 1열 (범주. 신경망·스태킹이 one-hot 으로 씀)
빼 둔 것: 생성 모델·SVM 커널·신경망 (모델. 신경망 입력은 생성 모델의 OOF 우도 = 타깃 사용), 섞기 가중치, 최근접 투표 (train 정답을
쓰는 후처리), 1.2제곱·평탄화, LR 성분의 희소 입력 (위치별 2–6-mer·중앙 쌍 one-hot + 1–3-gram 개수, 가중치 2%, k-mer 는 xgb 의 것과
겹침. 2026-11-18 사용자), one-hot 인코딩 (모델 입력). 타깃을 쓰는 피처는 위 것 말고는 없음.
train+test 로 fit 한 것: CountVectorizer 의 어휘 (어느 쪽에든 있는 n-gram 이 열이 됨, 값은 행마다).

동일성 확인 (로컬, 2026-11-18): 열 수가 에이전트 기록과 같음 (ext 의 one-hot + 수치 = 1,310, Xbase = 331, Xmotif = 2,496 의
one-hot 수준). 옮긴 열로 에이전트 ext-XGB 성분 (extrepeat.py, seed 11) 을 다시 돌려 extrepeat.log 의 RESULT 와 비교
(docs/tabarena.md 의 "C1"). 피처 배열은 작업 디렉터리에 없음.
TFM 피처 상한 없음 (1,165열. TabArena 의 FM 은 1,777열까지 돎).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.bench.tabarena.features import as_category
from fab.core.competition import load_module
from fab.core.features import split

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_ta_splice_f0")

TFM_MAX_FEATURES = None
EXT_GROUPS = ["A", "C", "G", "T", "CT", "AG", "CG", "AT"]
EXT_RUN_WINDOWS = [(0, 30), (5, 27), (10, 28), (20, 30), (30, 40), (30, 60)]
XBASE_ONLY = [(15, 28), (0, 60)]  # repeated.py 의 Xbase 창 11개 중 ext 의 창에 없는 것


def ext_numeric(a: np.ndarray, raw: pd.DataFrame) -> pd.DataFrame:
    """extended_trees.py 의 X 에서 one-hot 을 뺀 수치 열 그대로 (같은 순서)."""
    from sklearn.feature_extraction.text import CountVectorizer

    cols = {}
    for length in [5, 10, 15, 20, 30]:
        for begin in range(0, 61 - length, 5):
            for nt in EXT_GROUPS:
                cols[f"cnt_{nt}_{begin}_{begin + length}"] = np.isin(a[:, begin:begin + length], list(nt)).sum(axis=1)
    for begin, end in EXT_RUN_WINDOWS:
        seq = raw.iloc[:, begin:end].agg("".join, axis=1)
        for group in ["CT", "AG", "CG", "AT"]:
            c = np.isin(a[:, begin:end], list(group))
            run = np.zeros(len(raw))
            maxrun = np.zeros(len(raw))
            for i in range(end - begin):
                run = (run + 1) * c[:, i]
                maxrun = np.maximum(maxrun, run)
            cols[f"run_{group}_{begin}_{end}"] = maxrun
        cv = CountVectorizer(analyzer="char", ngram_range=(2, 3), dtype=np.float32)
        m = cv.fit_transform(seq).toarray()  # 어휘는 train+test (에이전트 그대로, 기본 lowercase)
        for j, gram in enumerate(cv.get_feature_names_out()):
            cols[f"gram_{gram}_{begin}_{end}"] = m[:, j]
    return pd.DataFrame(cols, index=raw.index).astype("float32")


def motif_cats(raw: pd.DataFrame) -> pd.DataFrame:
    """repeated.py 의 Xmotif 가 one-hot 한 위치별 2·3-mer 와 중앙 쌍 (raw 는 문자열)."""
    cols = {}
    for k in [2, 3]:
        for i in range(20, 40 - k + 1):
            cols[f"kmer{k}_{i}"] = raw.iloc[:, i:i + k].agg("".join, axis=1)
    for p in range(25, 35):
        for q in range(p + 2, 35):
            cols[f"pair_{p}_{q}"] = raw.iloc[:, p] + raw.iloc[:, q]
    return pd.DataFrame(cols, index=raw.index)


def build(train: pd.DataFrame, test: pd.DataFrame):
    X, X_test, cats = _f0.build(train, test)
    pos = list(test.columns[1:])  # id 다음의 60열 (에이전트의 te.columns[1:])
    raw = pd.concat([train[pos], test[pos]], keys=["train", "test"])
    a = raw.to_numpy()
    rs = raw.astype(str)
    num = ext_numeric(a, raw)
    for begin, end in XBASE_ONLY:
        for nt in "ACGT":
            num[f"cnt_{nt}_{begin}_{end}"] = np.sum(a[:, begin:end] == nt, axis=1).astype("float32")
    motif = motif_cats(rs)
    motif["motif_group"] = (((rs.iloc[:, 28] == "A") & (rs.iloc[:, 29] == "G")).astype(int)
                            + 2 * ((rs.iloc[:, 30] == "G") & (rs.iloc[:, 31] == "T")).astype(int)).astype(str)
    new = pd.concat([motif.apply(as_category), num], axis=1)
    new_tr, new_te = split(new)
    return (pd.concat([X, new_tr], axis=1), pd.concat([X_test, new_te], axis=1), [*cats, *motif.columns])
