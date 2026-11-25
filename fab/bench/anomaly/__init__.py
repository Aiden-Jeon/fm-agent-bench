"""표 이상탐지 트랙 (ADBench, 비지도): 대회 준비 (로컬), 기존 방법 · FM 제출, 채점 (원격 정리 작업), ADBench 공개값 비교, Elo.
조사 · 결정은 docs/anomaly.md.

    uv run --frozen fab ad init                                   # competitions/ad-<이름>/ (s1: config.yaml · description.md) 이 없으면 만듦
    uv run --frozen fab ad prepare all --splits all               # ADBench 분할 → 대회 data/ (public) + 정답 (FAB_ANOMALY)
    uv run --frozen fab ad fm all --methods iforest ecod ... --splits all     # CPU 작업 (pyod) + 채점 요청
    uv run --frozen fab ad fm all --methods tabpfn_unsup --splits s1          # GPU 작업 (TabPFN 비지도) + 채점 요청
    uv run --frozen fab ad compare                                # 우리 채점 대 ADBench Table D4 (results/anomaly_reference.csv)
    uv run --frozen fab ad elo                                    # IForest = 1000 → results/anomaly_elo.csv
    uv run --frozen fab ad agent run ad-cardio codex-gpt          # 에이전트 (s1, 익명 id 로)

대회 단위: (데이터셋, split) 마다 대회 하나. split = ADBench seed (s1·s2·s3 = seed 1·2·3). 저장소에는 s1
(competitions/ad-<이름>/: config.yaml · description.md) 만, s2·s3 은 그 설정을 본떠 만드는 competitions/ads-<이름>-s<k>/ (git 제외).

데이터: ADBench 저장소의 Classical npz (FAB_ANOMALY/src/adbench/adbench/datasets/Classical/, numpy 로 읽음 → 옆 가상환경 불필요).
FAB_ANOMALY (기본 <저장소>/../anomaly, worktree 에서는 절대경로로) 아래:
    src/adbench/          ADBench 저장소 (Minqi824/ADBench, BSD-2). git clone 후 npz 만 있으면 됨
    data/answers/         정답 (test 행의 id · label) + 정보 json. 로컬 캐시 (채점용 사본은 내 Volume 의 anomaly/answers/)
    data/train_labels/    train 행의 라벨: 깨끗한 context 진단 (+clean) 만 씀 (내 Volume 의 anomaly/train_labels/, FM 작업만 읽음)
    .venv/                로컬 확인용 (pyod · tabpfn-extensions). 이 프로젝트 uv.lock 은 안 바꿈
대회 data/ 에는 public 만 (train.csv · test.csv · sample_submission.csv · description.md). 라벨은 어느 파일에도 없음.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fab.core.competition import COMP_DIR, ROOT, Competition, load_competition
from fab.core.results import RESULTS, append_rows

# ADBench Classical 중 고른 12개 (docs/anomaly.md 1 의 규칙): 이름 (ADBench Table 의 이름) → npz 파일 이름 (번호_이름)
DATASETS = {"annthyroid": "2_annthyroid", "cardio": "6_cardio", "cover": "10_cover", "fault": "12_fault",
            "landsat": "19_landsat", "magic.gamma": "22_magic.gamma", "mammography": "23_mammography",
            "PageBlocks": "27_PageBlocks", "shuttle": "32_shuttle", "Waveform": "41_Waveform", "Wilt": "44_Wilt",
            "yeast": "47_yeast"}
PILOTS = ["cardio", "fault"]  # Agent 파일럿 (결정 3)
SPLITS = {"s1": 1, "s2": 2, "s3": 3}  # ADBench run.py 의 seed_list
AGENT_SPLIT = "s1"
AD_DIR = Path(os.environ.get("FAB_ANOMALY", ROOT.parent / "anomaly")).expanduser().resolve()
NPZ = AD_DIR / "src" / "adbench" / "adbench" / "datasets" / "Classical"
ANSWERS = AD_DIR / "data" / "answers"
TRAIN_LABELS = AD_DIR / "data" / "train_labels"
GRADES = RESULTS / "anomaly.csv"  # 우리 채점 (git 추적, 원격 정리 작업이 씀)
RUNS = RESULTS / "anomaly_runs.csv"  # 에이전트 실행별 채점 (git 추적, 원격 정리 작업이 씀. fab.agent 의 표)
ELO = RESULTS / "anomaly_elo.csv"  # IForest = 1000 Elo (git 추적, fab ad elo)
REFERENCE = RESULTS / "anomaly_reference.csv"  # ADBench 논문 Table D4 (AUROC) · D5 (AUPRC) 의 12개 행 (git 추적)
PREFIX, SPLIT_PREFIX = "ad-", "ads-"  # 저장소의 s1 대회 / 생성 split 대회 (git 제외)
ID_COL, SCORE_COL, LABEL_COL = "id", "score", "label"
# ADBench data_generator.py 의 분할 상수
TEST_SIZE, MAX_ROWS = 0.3, 10000
# 에이전트에게 보이면 안 되는 말 (데이터셋 이름 말고도): 벤치 · 원래 모음 이름
LEAK_WORDS = ["adbench", "odds", "anomaly detection benchmark"]


def slug(dataset: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")


def check_dataset(name: str) -> str:
    if name not in DATASETS:
        raise SystemExit(f"고른 12개에 없는 데이터셋: {name} (fab/bench/anomaly/__init__.py 의 DATASETS)")
    return name


# --- 대회 ------------------------------------------------------------------------------------------------------


def committed() -> dict[str, Competition]:
    """저장소의 s1 대회 (competitions/ad-*) {데이터셋: 대회}."""
    out = {}
    for p in sorted(COMP_DIR.glob(f"{PREFIX}*/config.yaml")):
        c = load_competition(p.parent.name)
        if c.anomaly:
            out[c.anomaly["dataset"]] = c
    return out


def comp_name(dataset: str, split: str) -> str:
    if split not in SPLITS:
        raise SystemExit(f"split 은 {list(SPLITS)} 중 하나: {split}")
    if split == AGENT_SPLIT and (c := committed().get(dataset)):
        return c.name
    return f"{SPLIT_PREFIX}{slug(dataset)}-{split}"


def resolve(target: str) -> str:
    """데이터셋 이름 · 대회 이름 → 데이터셋 이름."""
    if (COMP_DIR / target / "config.yaml").exists() and (ad := load_competition(target).anomaly):
        return ad["dataset"]
    return check_dataset(target)


def base_config(dataset: str) -> dict:
    """s1 대회 설정 (fab ad init 이 씀). anon_id 는 DATASETS 순서 (anomaly-p01 …)."""
    k = list(DATASETS).index(dataset) + 1
    return dict(name=f"{PREFIX}{slug(dataset)}",
                anomaly=dict(dataset=dataset, adbench=DATASETS[dataset], split=AGENT_SPLIT, seed=SPLITS[AGENT_SPLIT],
                             anon_id=f"anomaly-p{k:02d}"),
                task="anomaly", target=LABEL_COL, id_col=ID_COL, metric="auroc", cv={})


def split_comp(dataset: str, split: str) -> Competition:
    """(데이터셋, split) 대회. s1 은 저장소의 대회, s2·s3 은 그 설정에서 이름 · split · seed 만 바꾸고 익명 id 를 뺀 생성 폴더."""
    name = comp_name(dataset, split)
    if not name.startswith(SPLIT_PREFIX):
        return load_competition(name)
    base = committed().get(dataset)
    cfg = yaml.safe_load((base.dir / "config.yaml").read_text()) if base else base_config(dataset)
    cfg["anomaly"] = {k: v for k, v in cfg["anomaly"].items() if k != "anon_id"}
    cfg["name"], cfg["anomaly"]["split"], cfg["anomaly"]["seed"] = name, split, SPLITS[split]
    d = COMP_DIR / name
    d.mkdir(exist_ok=True)
    text = "# 생성 파일 (fab/bench/anomaly/__init__.py 의 split_comp, git 제외). 고치려면 s1 설정을 고치고 다시 만듦\n" + \
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if not (d / "config.yaml").exists() or (d / "config.yaml").read_text() != text:
        (d / "config.yaml").write_text(text)
    return load_competition(name)


# --- 데이터 · 분할 (ADBench data_generator.py 그대로) ------------------------------------------------------------


@cache
def load_npz(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    f = NPZ / f"{DATASETS[check_dataset(dataset)]}.npz"
    if not f.exists():
        raise SystemExit(f"{f} 없음 → ADBench 저장소를 FAB_ANOMALY/src/adbench 에 (fab/bench/anomaly/__init__.py 위쪽 설명)")
    d = np.load(f, allow_pickle=False)  # 받은 파일: pickle 안 풂
    return np.asarray(d["X"], dtype=float), np.asarray(d["y"]).astype(int)


def adbench_split(X: np.ndarray, y: np.ndarray, seed: int):
    """ADBench DataGenerator.generator (비지도, la=0, 잡음 없음) 의 분할 → (X_train, X_test, y_train, y_test), MinMax 는 train 으로.
    순서도 원문과 같게: set_seed → (1만 행 넘으면 set_seed 다시 → 1만 행 비복원 표본) → train_test_split (random_state 없음 =
    numpy 전역 상태). 1,000 행 미만의 복원 추출 불리기는 그런 데이터셋을 고르지 않아 (docs/anomaly.md 1 의 b) 여기 없음."""
    import random

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    if len(y) < 1000:
        raise ValueError(f"행 {len(y)} < 1000: ADBench 는 복원 추출로 불림 → 이 트랙에서는 안 씀")
    np.random.seed(seed)
    random.seed(seed)
    if len(y) > MAX_ROWS:
        np.random.seed(seed)
        random.seed(seed)
        idx = np.random.choice(np.arange(len(y)), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, stratify=y)
    scaler = MinMaxScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def feature_names(d: int) -> list[str]:
    return [f"f{j}" for j in range(d)]


def prepare(comp: Competition):
    """fab download 가 부름: split 의 public → comp.data_dir, 정답 → ANSWERS/<대회>.{csv,json}, train 라벨 → TRAIN_LABELS.
    행 id = (ADBench 번호, seed) 고정 seed 의 무작위 순열을 train · test 가 나눠 가짐 (id 에 정보 없음).
    행 순서는 ADBench 분할이 내놓은 순서 그대로 (train_test_split 이 섞은 무작위 순서라 라벨 정보 없음). TabArena 처럼 id 순으로
    다시 놓으면 IForest 처럼 행 순서로 표본을 뽑는 방법이 ADBench 와 달라짐 (2026-11-25 로컬: cardio IForest 0.9315 대 0.9319)."""
    ad = comp.anomaly
    X, y = load_npz(ad["dataset"])
    X_train, X_test, y_train, y_test = adbench_split(X, y, ad["seed"])
    cols = feature_names(X.shape[1])
    num = int(DATASETS[ad["dataset"]].split("_", 1)[0])
    ids = np.random.default_rng([num, ad["seed"]]).permutation(len(y_train) + len(y_test))
    train = pd.DataFrame(X_train, columns=cols).assign(**{ID_COL: ids[:len(y_train)], LABEL_COL: y_train})
    test = pd.DataFrame(X_test, columns=cols).assign(**{ID_COL: ids[len(y_train):], LABEL_COL: y_test})
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    train[[ID_COL, *cols]].to_csv(comp.data_dir / "train.csv", index=False)  # 값은 float repr (다시 읽으면 같은 값)
    test[[ID_COL, *cols]].to_csv(comp.data_dir / "test.csv", index=False)
    test[[ID_COL]].assign(**{SCORE_COL: 0.0}).to_csv(comp.data_dir / "sample_submission.csv", index=False)
    if (comp.dir / "description.md").exists():
        shutil.copy(comp.dir / "description.md", comp.data_dir / "description.md")
    for d in (ANSWERS, TRAIN_LABELS):
        d.mkdir(parents=True, exist_ok=True)
    test[[ID_COL, LABEL_COL]].to_csv(ANSWERS / f"{comp.name}.csv", index=False)
    train[[ID_COL, LABEL_COL]].to_csv(TRAIN_LABELS / f"{comp.name}.csv", index=False)
    info = dict(comp=comp.name, dataset=ad["dataset"], adbench=ad["adbench"], split=ad["split"], seed=ad["seed"],
                metric="auroc", n_train=len(train), n_test=len(test), d=len(cols),
                train_anomalies=int(y_train.sum()), test_anomalies=int(y_test.sum()))
    (ANSWERS / f"{comp.name}.json").write_text(json.dumps(info, indent=1))
    print(f"{comp.name}: train {len(train)} · test {len(test)} (이상 {info['test_anomalies']}) · 피처 {len(cols)} → "
          f"{comp.data_dir}")


def ensure_prepared(comp: Competition):
    """없으면 prepare. 있으면 description.md 만 대회 폴더의 것으로 맞춤."""
    if not ((comp.data_dir / "train.csv").exists() and (ANSWERS / f"{comp.name}.json").exists()
            and (TRAIN_LABELS / f"{comp.name}.csv").exists()):
        return prepare(comp)
    src, dst = comp.dir / "description.md", comp.data_dir / "description.md"
    if src.exists() and (not dst.exists() or dst.read_bytes() != src.read_bytes()):
        shutil.copy(src, dst)
        print(f"{comp.name}: description.md 갱신")


def make_comps(dataset: str, splits: list[str]) -> list[Competition]:
    comps = [split_comp(dataset, s) for s in splits]
    for c in comps:
        ensure_prepared(c)
    return comps


def pick_splits(spec: str) -> list[str]:
    if spec == "all":
        return list(SPLITS)
    if bad := [s for s in spec.split(",") if s not in SPLITS]:
        raise SystemExit(f"없는 split: {bad} (있는 것: {list(SPLITS)})")
    return spec.split(",")


# --- s1 대회 폴더 (fab ad init) ---------------------------------------------------------------------------------

DESCRIPTION = """\
# Unsupervised anomaly detection on a numeric table

## Description

Each row is one observation described by {d} numeric features. A minority of the rows are anomalies, but no row is labeled.
`train.csv` is an unlabeled sample of {n_train} rows; it contains anomalies at about the same (unknown) rate as the test set.
Your task is to give every row of `test.csv` an anomaly score: the higher the score, the more likely the row is an anomaly.

The features have no names and no documented meaning. Each feature was min-max scaled with the minimum and maximum of
`train.csv`, so test values can fall slightly outside [0, 1]. There are no missing values. Rows are independent and given
in random order; the `id` column is a random identifier and carries no information.

## Rules

- There are no labels anywhere: build an unsupervised detector.
- Fit everything (preprocessing, feature transforms, models, hyperparameters) on the rows of `train.csv` only. Do not fit,
  tune or transform anything using the rows of `test.csv`; score each test row with what you fitted on `train.csv`, as if
  the test rows arrived one at a time after training.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUROC) between the submitted score and the hidden anomaly label
of each test row. Only the ordering of the scores matters; any finite real number is a valid score.

## Submission File

For each `id` in the test set, give an anomaly score (higher = more anomalous). The file should contain a header and have
the following format:

```
id,score
{id0},0.0
{id1},0.0
...
```

## Dataset Description

- `train.csv` — the unlabeled training sample: `id` and the {d} features `f0`, ..., `f{last}` ({n_train} rows)
- `test.csv` — the rows to score: `id` and the same {d} features ({n_test} rows)
- `sample_submission.csv` — a sample submission file in the correct format
"""


def init_committed():
    """competitions/ad-<이름>/ (s1) 의 config.yaml · description.md 를 없으면 만듦 (있으면 그대로: 고친 것을 덮지 않음).
    설명의 행 수 · 첫 id 는 s1 분할을 만들어 봐서 채움 (prepare 와 같은 계산)."""
    for ds in DATASETS:
        cfg = base_config(ds)
        d = COMP_DIR / cfg["name"]
        d.mkdir(exist_ok=True)
        if not (d / "config.yaml").exists():
            head = (f"# ADBench {DATASETS[ds]} 의 s1 split (seed 1). 다른 split 은 fab ad prepare --splits all 이 이 설정을 본떠\n"
                    f"# competitions/{SPLIT_PREFIX}<이름>-s<k>/ 로 만듦 (git 제외). anon_id = 에이전트에게 보이는 이름 (docs/anomaly.md 6)\n")
            (d / "config.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        if not (d / "description.md").exists():
            X, y = load_npz(ds)
            X_train, X_test, _, _ = adbench_split(X, y, SPLITS[AGENT_SPLIT])
            num = int(DATASETS[ds].split("_", 1)[0])
            ids = np.random.default_rng([num, SPLITS[AGENT_SPLIT]]).permutation(len(X_train) + len(X_test))
            first = ids[len(X_train):][:2]  # test.csv 의 첫 두 행 (prepare 와 같은 순서)
            (d / "description.md").write_text(DESCRIPTION.format(d=X.shape[1], last=X.shape[1] - 1, n_train=len(X_train),
                                                                 n_test=len(X_test), id0=first[0], id1=first[1]))
        print(f"{d.name}: {cfg['anomaly']['anon_id']}")


def leak_terms(comp: Competition) -> list[str]:
    """에이전트에게 보이면 안 되는 글 (fab/agent 의 leak_check): 데이터셋 이름 (원래 · 소문자 · 구분자 바꾼 것), ADBench 파일
    이름, 대회 이름, 벤치 이름."""
    ds = comp.anomaly["dataset"].lower()
    words = {ds, ds.replace(".", " "), ds.replace(".", "_"), ds.replace(".", "-"), comp.anomaly["adbench"].lower(),
             comp.name, *LEAK_WORDS}
    return sorted(words)


def agent_target(name: str) -> tuple[Competition, str]:
    """에이전트 대회 (s1, 설정에 anon_id) → (대회, 익명 id)."""
    comp = load_competition(name)
    ad = comp.anomaly or {}
    if not ad.get("anon_id"):
        raise SystemExit(f"{name}: 설정에 anomaly.anon_id 가 없음 (에이전트는 s1 의 {PREFIX}<이름> 대회만)")
    if ad.get("split") != AGENT_SPLIT:
        raise SystemExit(f"{name}: 에이전트는 {AGENT_SPLIT} 만 ({ad.get('split')})")
    if not (comp.dir / "description.md").exists():
        raise SystemExit(f"{comp.dir}/description.md 없음")
    return comp, ad["anon_id"]


# --- 채점 ------------------------------------------------------------------------------------------------------


def read_answers(answers_csv: Path) -> pd.DataFrame:
    return pd.read_csv(answers_csv)


def score_frame(sub: pd.DataFrame, ans: pd.DataFrame) -> dict:
    """제출 표 (id, score) → {score (AUROC), metric_error (1 − AUROC), aupr, valid_submission} 또는 무효 + error."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if list(map(str, sub.columns)) != [ID_COL, SCORE_COL]:
        return dict(valid_submission=False, error=f"열 {list(sub.columns)} ≠ {[ID_COL, SCORE_COL]}")
    if len(sub) != len(ans) or set(sub[ID_COL]) != set(ans[ID_COL]) or sub[ID_COL].duplicated().any():
        return dict(valid_submission=False, error=f"행·id 가 test 와 다름 ({len(sub)}행, test {len(ans)}행)")
    s = sub.set_index(ID_COL).loc[ans[ID_COL], SCORE_COL]
    try:
        s = s.to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        return dict(valid_submission=False, error=f"점수가 숫자가 아님: {e!r}"[:300])
    if not np.isfinite(s).all():
        return dict(valid_submission=False, error="점수에 NaN/inf")
    y = ans[LABEL_COL].to_numpy(dtype=int)
    auroc = float(roc_auc_score(y, s))
    return dict(valid_submission=True, score=auroc, metric_error=1 - auroc, aupr=float(average_precision_score(y, s)))


def score_submission(sub_csv: Path, answers_csv: Path) -> dict:
    try:
        sub = pd.read_csv(sub_csv)
    except Exception as e:  # noqa: BLE001
        return dict(valid_submission=False, error=f"읽기 실패: {e!r}")
    return score_frame(sub, read_answers(answers_csv))


def grade_row(comp: Competition, model: str, report: dict, note: str, info: dict) -> dict:
    ad = comp.anomaly
    return dict(timestamp=pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds"), comp=comp.name,
                dataset=ad["dataset"], split=ad["split"], seed=ad["seed"], model=model, metric="auroc",
                score=report.get("score"), metric_error=report.get("metric_error"), aupr=report.get("aupr"),
                run=info.get("run"), note=note, opts=info.get("opts"))


def grade(comps: list[str], models: list[str], note: str, grader, timer: dict | None = None) -> tuple[list[dict], dict]:
    """(대회, 방법) 마다 test.npy (test.csv 행 순서의 점수) → 제출 표 → score_frame → results/anomaly.csv.
    원격 정리 작업이 부름 (grader = remote.Grader). 예측이 없거나 채점에 실패한 것은 표에 안 씀 (errors 로).
    TabArena 의 grade 와 같은 꼴 (reconcile 의 grade_split_bench 가 같이 부름)."""
    from fab.core.results import OUTPUTS, test_info

    rows, errors = [], {}
    for name in comps:
        comp = load_competition(name)
        infos = {m: test_info(name, m) for m in models}
        for m in [m for m in models if infos[m] is None]:
            errors[f"{name}/{m}"] = "test 예측 없음"
        if not (todo := [m for m in models if infos[m] is not None]):
            continue
        try:
            ans = grader.answers(name)
            ids = pd.read_csv(comp.data_dir / comp.test_file, usecols=[ID_COL])[ID_COL]
        except Exception as e:  # noqa: BLE001
            errors |= {f"{name}/{m}": f"채점 실패: {e!r}"[:500] for m in todo}
            continue
        for m in todo:
            try:
                s = np.load(OUTPUTS / name / m / "test.npy")
                report = score_frame(pd.DataFrame({ID_COL: ids, SCORE_COL: s}), ans)
            except Exception as e:  # noqa: BLE001
                errors[f"{name}/{m}"] = f"채점 실패: {e!r}"[:500]
                continue
            if not report.get("valid_submission"):
                errors[f"{name}/{m}"] = f"채점 실패: {report.get('error')}"
                continue
            rows.append(grade_row(comp, m, report, note, infos[m]))
    if rows:
        append_rows(pd.DataFrame(rows), GRADES)
    return rows, errors


def request_grade(comps: list[str], models: list[str], note: str = ""):
    """채점 요청 → 원격 정리 작업이 그 방법의 실행이 끝나기를 기다렸다가 채점 (TabArena 와 같은 흐름)."""
    from datetime import datetime

    from fab.bench.anomaly.remote import upload_answers
    from fab.remote import settings, submit_item

    cfg = settings()
    upload_answers(cfg, comps)
    key = f"anomaly__{datetime.now():%Y%m%d-%H%M%S-%f}"
    submit_item(cfg, "grade", key, dict(bench="anomaly", comps=comps, models=models, note=note))
    print(f"채점 요청: {key} (대회 {len(comps)}개 × 방법 {len(models)}개) → 실행이 끝난 뒤 fab remote kick, 결과는 fab sync")


# --- 기존 방법 · FM 제출 ----------------------------------------------------------------------------------------


def check_methods(methods: list[str]):
    from fab.bench.anomaly.methods import FM, PYOD, is_known

    if bad := [m for m in methods if not is_known(m)]:
        raise SystemExit(f"알 수 없는 방법: {bad}. 있는 것: {', '.join([*PYOD, *FM])} (+clean 을 붙이면 깨끗한 context 진단)")


def submit_fm(todo: list[tuple[str, list[str]]], methods: list[str], note: str, timeout: int, accelerator: str | None,
              grade_after: bool = True) -> list[dict]:
    """데이터셋 (이름, split) 들의 대회를 CPU 작업 하나 (pyod) + GPU 작업 하나 (TabPFN) 로. 작업 안에서 대회마다 `fab run`
    (fab/runner/databricks.py 의 RUNNER 그대로 → fab/runner/run.py 가 anomaly 대회는 run.py 로 넘김). 결과는 정리 작업의 fm 단계가 합침."""
    from fab.bench.anomaly.methods import is_clean, is_gpu
    from fab.bench.anomaly.remote import train_labels_dir, upload_train_labels
    from fab.bench.tabarena.remote import upload_data
    from fab.runner.databricks import record, submit, submit_cpu
    from fab.runner.databricks import settings as dbx_settings

    check_methods(methods)
    comps = [c for ds, splits in todo for c in make_comps(ds, splits)]
    cfg = dbx_settings(argparse.Namespace(accelerator=accelerator, image=None))
    gpu = [m for m in methods if is_gpu(m)]
    cpu = [m for m in methods if not is_gpu(m)]
    extra = []
    if any(is_clean(m) for m in methods):  # 깨끗한 context 진단: train 라벨 (내 Volume, 에이전트 SP 는 못 읽음)
        upload_train_labels(cfg, [c.name for c in comps])
        extra += ["--train-labels", train_labels_dir(cfg)]
    data = upload_data(cfg, comps)
    label = f"ad-{slug(todo[0][0])}-{len(comps)}splits" if len(todo) == 1 else f"ad-{len(todo)}datasets-{len(comps)}splits"
    runs = []
    for ms, is_cpu in ((cpu, True), (gpu, False)):
        if not ms:
            continue
        jobs = [(c.name, data[c.name], [*ms, "--note", note, *extra]) for c in comps]
        if is_cpu:
            run = submit_cpu(cfg, label, None, None, ms, timeout, "", jobs=jobs)
        else:
            run = submit(cfg, label, None, None, ms, cfg["accelerator"], timeout, "", jobs=jobs)
        record(cfg, run)
        runs.append(run)
    if grade_after:
        request_grade([c.name for c in comps], methods, note)
    return runs


# --- 비교 · Elo -------------------------------------------------------------------------------------------------


def latest_grades(grades: pd.DataFrame) -> pd.DataFrame:
    return grades.sort_values("timestamp", kind="stable").drop_duplicates(["comp", "model"], keep="last")


def compare(tol: float = 0.005) -> pd.DataFrame:
    """우리 채점 (seed 3개 평균) 대 ADBench Table D4 (같은 방법, seed 1~3 평균, pyod 1.0.0). split 3개가 다 있는 것만 판정."""
    from fab.bench.anomaly.methods import CLEAN

    ours = latest_grades(pd.read_csv(GRADES))
    ours = ours[~ours.model.str.endswith(CLEAN)]
    ref = pd.read_csv(REFERENCE)
    g = ours.groupby(["dataset", "model"]).agg(ours=("score", "mean"), n=("split", "nunique"), aupr=("aupr", "mean"))
    out = g.reset_index().merge(ref.rename(columns={"method": "model", "aucroc": "adbench", "aucpr": "adbench_aupr"})
                                [["dataset", "model", "adbench", "adbench_aupr"]], on=["dataset", "model"], how="left")
    out["diff"] = out.ours - out.adbench
    out["within"] = (out["diff"].abs() <= tol).where(out.n.eq(len(SPLITS)) & out.adbench.notna())
    return out


def agent_runs(runs: pd.DataFrame, grades: pd.DataFrame) -> pd.DataFrame:
    """Agent 실행 → 회차마다 method · dataset · split · metric_error · failed. 실패한 회차 (무효 · 점수 없음) 는 그 (데이터셋,
    split) 의 기준점 (IForest) 값 (TabArena 의 RandomForest 채우기와 같음). +fm · +gw 설정은 뺌 (TabArena 와 같은 규칙)."""
    from fab.agent import FM_SUFFIX, GATEWAY_SUFFIX
    from fab.bench.anomaly.methods import ANCHOR

    runs = runs[~(runs.config.str.contains(FM_SUFFIX, regex=False) | runs.config.str.contains(GATEWAY_SUFFIX, regex=False))]
    dataset_of = {c.name: ds for ds, c in committed().items()}
    r = runs.assign(dataset=runs.comp.map(dataset_of), split=AGENT_SPLIT, method="agent: " + runs.config)
    r = r[r.dataset.notna()].copy()
    r["failed"] = ~r.valid_submission.eq(True) | r.metric_error.isna()
    anchor = grades[grades.model == ANCHOR].set_index(["dataset", "split"]).metric_error
    r["metric_error"] = r.metric_error.where(~r.failed, pd.Series(list(zip(r.dataset, r.split)), index=r.index).map(anchor))
    return r[["method", "dataset", "split", "metric_error", "failed"]]


def elo_tables(rounds: int) -> pd.DataFrame:
    """두 풀 → results/anomaly_elo.csv. fm: 12개 × split 3개 (기존 10개 + FM), agent: Agent 를 돌린 데이터셋의 s1 (+ Agent).
    +clean 진단은 안 넣음 (train 라벨을 씀). 기준점 IForest = 1000 (elo.py 의 elo(anchor=...))."""
    from fab.bench.anomaly.methods import ANCHOR, CLEAN, FM, PYOD
    from fab.bench.tabarena import elo

    grades = latest_grades(pd.read_csv(GRADES))
    grades = grades[grades.model.isin([*PYOD, *FM]) & ~grades.model.str.endswith(CLEAN) & grades.metric_error.notna()]
    long = grades[["model", "dataset", "split", "metric_error"]].rename(columns={"model": "method"})
    if missing := sorted(set(DATASETS) - set(long[long.method == ANCHOR].dataset)):
        print(f"경고: 기준점 {ANCHOR} 가 없는 데이터셋 {missing} → 그 데이터셋은 풀에서 빠짐")
    pools = [("fm", elo.fill(long, list(dict.fromkeys(long.method)), anchor=ANCHOR))]
    counts = None
    if RUNS.exists() and len(runs := pd.read_csv(RUNS)):
        ar = agent_runs(runs, grades)
        agents = (ar.groupby(["method", "dataset", "split"], as_index=False)
                  .agg(metric_error=("metric_error", "mean"), imputed=("failed", "mean")))
        keep = set(agents.dataset)
        base = long[long.dataset.isin(keep) & (long.split == AGENT_SPLIT)]
        both = pd.concat([base, agents], ignore_index=True)
        pools.append(("agent", elo.fill(both, list(dict.fromkeys(both.method)), anchor=ANCHOR)))
        counts = ar.groupby(["method", "dataset"]).size().groupby("method").mean()
    out = []
    for pool, lg in pools:
        t = elo.elo(lg, rounds=rounds, anchor=ANCHOR)
        if pool == "agent":
            t["runs"] = counts
        n = int(t.datasets.iloc[0])
        if n < elo.MIN_CI_DATASETS:
            print(f"주의: {pool} 풀의 데이터셋 {n}개 (< {elo.MIN_CI_DATASETS}) → 95% 구간은 비움")
        out.append(t.reset_index().assign(pool=pool))
        with pd.option_context("display.width", 200):
            print(f"\n== {pool}: 데이터셋 {n}개, (데이터셋, split) {lg[['dataset', 'split']].drop_duplicates().shape[0]}개")
            print(t.round({"elo": 0, "elo+": 0, "elo-": 0, "rank": 2, "winrate": 3, "imputed": 3}).to_string())
    res = pd.concat(out, ignore_index=True)
    cols = ["pool", "method", "elo", "elo+", "elo-", "rank", "winrate", "imputed", "datasets"]
    res = res[cols + (["runs"] if "runs" in res else [])]
    res.round({"elo": 1, "elo+": 1, "elo-": 1, "rank": 3, "winrate": 4, "imputed": 4}).to_csv(ELO, index=False)
    print(f"\n{ELO}: {len(res)}행")
    return res


# --- 명령 ------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    argv = sys.argv[2:] if argv is None else argv
    if argv[:1] == ["agent"]:
        from fab.agent import main as agent_main

        return agent_main(argv[1:], bench="anomaly")
    ap = argparse.ArgumentParser(prog="fab ad", description="표 이상탐지 (ADBench, docs/anomaly.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="competitions/ad-<이름>/ (s1) 의 config.yaml · description.md 를 없으면 만듦")
    p = sub.add_parser("prepare", help="ADBench 분할 → 대회 data/ (public), 정답 · train 라벨은 FAB_ANOMALY")
    p.add_argument("datasets", nargs="+", help="데이터셋 또는 대회 이름, all = 12개, pilots = 파일럿 2개")
    p.add_argument("--splits", default="s1", help="all 또는 s1,s2 처럼 (기본 s1)")
    f = sub.add_parser("fm", help="기존 방법 (CPU 작업) · FM (GPU 작업) 제출 + 채점 요청")
    f.add_argument("datasets", nargs="+")
    f.add_argument("--methods", nargs="+", required=True)
    f.add_argument("--splits", default="all")
    f.add_argument("--note", default="")
    f.add_argument("--timeout", type=int, default=240, help="작업 시간 제한 (분)")
    f.add_argument("--accelerator", default=None, help="GPU 작업 (기본: databricks.local.yaml 의 accelerator)")
    f.add_argument("--no-grade", action="store_true")
    g = sub.add_parser("grade", help="채점 요청 (원격 정리 작업) → fab sync 로 results/anomaly.csv")
    g.add_argument("datasets", nargs="+")
    g.add_argument("--methods", nargs="+", required=True)
    g.add_argument("--splits", default="all")
    g.add_argument("--note", default="")
    c = sub.add_parser("compare", help="우리 채점 (seed 평균) 대 ADBench Table D4")
    c.add_argument("--tol", type=float, default=0.005)
    e = sub.add_parser("elo", help="IForest = 1000 Elo: fm 풀 (12개 × 3 split) · agent 풀 (s1) → results/anomaly_elo.csv")
    e.add_argument("--rounds", type=int, default=200)
    sub.add_parser("agent", help="에이전트 실행 (fab ad agent -h, fab ta agent 와 같은 명령)")
    args = ap.parse_args(argv)

    def names(xs):
        if xs == ["all"]:
            return list(DATASETS)
        if xs == ["pilots"]:
            return list(PILOTS)
        return list(dict.fromkeys(resolve(x) for x in xs))

    if args.cmd == "init":
        init_committed()
    elif args.cmd == "prepare":
        for ds in names(args.datasets):
            make_comps(ds, pick_splits(args.splits))
    elif args.cmd == "fm":
        todo = [(ds, pick_splits(args.splits)) for ds in names(args.datasets)]
        submit_fm(todo, args.methods, args.note, args.timeout, args.accelerator, not args.no_grade)
        print("끝난 뒤 fab remote kick → 결과 합치기 · 채점, fab sync 로 results/anomaly.csv")
    elif args.cmd == "grade":
        check_methods(args.methods)
        comps = [c.name for ds in names(args.datasets) for c in make_comps(ds, pick_splits(args.splits))]
        request_grade(comps, args.methods, args.note)
    elif args.cmd == "compare":
        cmp = compare(args.tol)
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(cmp.round(4).to_string(index=False))
            judged = cmp[cmp.within.notna()]
            print("\n" + judged.groupby("model").within.agg(["sum", "count"]).rename(
                columns={"sum": f"|차| ≤ {args.tol}", "count": "데이터셋"}).to_string())
    elif args.cmd == "elo":
        elo_tables(args.rounds)

