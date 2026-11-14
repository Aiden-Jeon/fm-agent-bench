"""TabArena-v0.1 연동: OpenML 고정 분할로 split 별 대회 준비 (로컬), 채점 (원격 정리 작업, remote.py·fab/remote),
TabArena 공개 결과와 비교. 조사·결정은 docs/tabarena.md.

    uv run --frozen fab ta prepare hazelnut-spread-contaminant-detection --splits all   # 데이터·분할 받기 → split 대회
    uv run --frozen fab ta fm <데이터셋...> --models tabpfn_v2 tabicl kumo lgbm catboost --splits all  # 제출 + 채점 요청
    uv run --frozen fab ta reference                     # TabArena 공개 결과 (기본 설정) → results/tabarena_reference.csv
    uv run --frozen fab ta compare                       # results/tabarena.csv 와 TabArena 결과 비교 (맞춤 기준)
    uv run --frozen fab ta agent run ta-<데이터셋> aide-sonnet   # 에이전트 (r0f0, 익명 id 로)

대회 단위 (결정 5): (데이터셋, split) 마다 대회 하나. split = OpenML task 의 (repeat r, fold f) → 이름 r<r>f<f>,
TabArena 결과의 fold 번호 = 3r + f. 저장소에는 r0f0 (competitions/ta-<데이터셋>/: config.yaml·description.md·features.py) 만,
다른 split 은 split_comp 가 그 설정을 본떠 만드는 competitions/tas-<데이터셋>-r<r>f<f>/ (git 제외). r0f0 폴더가 없는
데이터셋 (FM 맞춤 확인용 51개) 은 r0f0 도 tas- 로 만듦.

데이터: openml·pyarrow 가 이 프로젝트 uv 환경에 없음 (uv.lock 을 안 바꿈) → 옆 가상환경에서 driver.py 를 `-I` 로:

    uv venv --python 3.12 $FAB_TABARENA/.venv
    uv pip install --python $FAB_TABARENA/.venv/bin/python openml pyarrow pandas

FAB_TABARENA (기본 ../tabarena) 아래: .venv/, data/cache/<task_id>/ (driver fetch 결과), data/answers/<대회>.{csv,json} (정답).
대회 data/ 에는 public 만 (train.csv·test.csv·sample_submission.csv·description.md). 행 id 는 split 마다 새로 매기고
(무작위 순열, OpenML 행 번호를 주지 않음) 행 순서도 섞음. 정답은 로컬 캐시와 채점용 내 Volume 의 tabarena/answers/ 에만
(에이전트 SP 의 scratch Volume 에는 public 만, remote.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fab.core.competition import COMP_DIR, ROOT, Competition, load_competition
from fab.core.results import RESULTS, append_rows

# OpenML 스위트 457 (tabarena-v0.1)
DATASETS = Path(__file__).with_name("datasets.csv")  # TabArena 메타데이터 (Apache-2.0) 중 필요한 열, 51개
TA_DIR = Path(os.environ.get("FAB_TABARENA", ROOT.parent / "tabarena")).expanduser().resolve()
CACHE = TA_DIR / "data" / "cache"
ANSWERS = TA_DIR / "data" / "answers"
DRIVER = Path(__file__).with_name("driver.py")
GRADES = RESULTS / "tabarena.csv"  # 우리 split 채점 (git 추적, 원격 정리 작업이 씀)
REFERENCE = RESULTS / "tabarena_reference.csv"  # TabArena 공개 결과의 기본 설정 행 (git 추적, fab ta reference)
RUNS = RESULTS / "tabarena_runs.csv"  # 에이전트 실행별 채점 (git 추적, 원격 정리 작업이 씀. fab.agent.TA_TABLE)
SPLIT_PREFIX = "tas-"  # 생성 split 대회 (git 제외)
ID_COL = "id"
METRICS = {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}
# 공개 결과 parquet (docs/tabarena.md 의 "출처"): (TabArena 이름, 묶음, 받는 곳)
S3, DATA = "https://tabarena.s3.amazonaws.com/cache/artifacts", "https://data.tabarena.ai/cache/artifacts"
REFERENCE_METHODS = [("TabPFNv2_GPU", "tabarena-2025-06-12", S3), ("TabICL_GPU", "tabarena-2025-06-12", S3),
                     ("LightGBM", "tabarena-2025-06-12", S3), ("RandomForest", "tabarena-2025-06-12", S3),
                     ("CatBoost", "tabarena-2026-07-13", DATA), ("TabICLv2", "tabarena-2026-09-16", DATA),
                     ("Kumo-Tabular", "tabarena-2026-10-06", DATA)]
# 2025-06-12 묶음의 TFM 은 못 도는 데이터셋 행도 있고 값이 RandomForest 대체값 → 메타데이터 can_run_* 가 False 인 행을 뺌
CAN_RUN = {"TabPFNv2_GPU": "can_run_tabpfnv2", "TabICL_GPU": "can_run_tabicl"}
# 우리 모델 → 비교할 TabArena 방법 (docs/tabarena.md 3 의 표)
COUNTERPART = {"tabpfn_v2": "TabPFNv2_GPU", "tabicl": "TabICLv2", "kumo": "Kumo-Tabular", "kumo_e16": "Kumo-Tabular",
               "lgbm": "LightGBM", "catboost": "CatBoost"}


# --- 데이터셋·split ------------------------------------------------------------------------------------------


@cache
def datasets() -> pd.DataFrame:
    return pd.read_csv(DATASETS).set_index("dataset")


def dataset_row(name: str) -> pd.Series:
    ds = datasets()
    if name not in ds.index:
        raise SystemExit(f"TabArena-v0.1 에 없는 데이터셋: {name} (fab/bench/tabarena/datasets.csv)")
    return ds.loc[name]


def slug(dataset: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")


def split_names(dataset: str) -> list[str]:
    """TabArena 가 쓰는 split (행 < 2,500 이면 반복 10번, 아니면 3번) × 3 fold."""
    return [f"r{r}f{f}" for r in range(int(dataset_row(dataset)["n_repeats"])) for f in range(3)]


def parse_split(split: str) -> tuple[int, int]:
    m = re.fullmatch(r"r(\d+)f(\d+)", split)
    if not m:
        raise ValueError(f"split 이름 r<repeat>f<fold> 아님: {split}")
    return int(m[1]), int(m[2])


def fold_index(split: str) -> int:
    """TabArena 결과의 fold 열 = 3 × repeat + fold."""
    r, f = parse_split(split)
    return 3 * r + f


def committed() -> dict[str, Competition]:
    """저장소의 r0f0 대회 (competitions/ta-*) {데이터셋: 대회}."""
    out = {}
    for p in sorted(COMP_DIR.glob("ta-*/config.yaml")):
        c = load_competition(p.parent.name)
        if c.tabarena:
            out[c.tabarena["dataset"]] = c
    return out


def comp_name(dataset: str, split: str) -> str:
    if split == "r0f0" and (c := committed().get(dataset)):
        return c.name
    return f"{SPLIT_PREFIX}{slug(dataset)}-{split}"


def resolve(target: str) -> str:
    """데이터셋 이름 또는 대회 이름 → 데이터셋 이름."""
    if (COMP_DIR / target / "config.yaml").exists() and (ta := load_competition(target).tabarena):
        return ta["dataset"]
    dataset_row(target)
    return target


# --- 받기 (driver) ---------------------------------------------------------------------------------------------


def _python() -> Path:
    py = TA_DIR / ".venv" / "bin" / "python"
    if not py.exists():
        sys.exit(f"TabArena 가상환경 없음: {py}\n설치 방법은 fab/bench/tabarena/__init__.py 위쪽 설명 (FAB_TABARENA 로 위치 지정)")
    return py


def _driver(*args: str) -> dict:
    # -I: 받은 파일이 있는 곳의 모듈을 import 하지 않게. cwd 도 받은 데이터 밖 (임시 폴더)
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(_python()), "-I", str(DRIVER), *args], stdout=subprocess.PIPE, text=True, check=True,
                             cwd=d)
    return json.loads(out.stdout.strip().splitlines()[-1])


def fetched(dataset: str) -> Path:
    """OpenML 데이터·분할 (한 번만 받음) → CACHE/<task_id>/."""
    tid = int(dataset_row(dataset)["task_id"])
    d = CACHE / str(tid)
    if not (d / "meta.json").exists():
        print(f"OpenML task {tid} ({dataset}) 받기 → {d}")
        _driver("fetch", str(tid), str(d))
    meta = json.loads((d / "meta.json").read_text())
    if meta["dataset"] != dataset:
        raise SystemExit(f"{d}: 데이터셋 이름이 다름 ({meta['dataset']} ≠ {dataset})")
    return d


def categorical(meta: dict) -> list[str]:
    """범주형 열: OpenML 의 범주형 표시 + 숫자가 아닌 열 (in_vehicle 의 coupon 은 OpenML 에서 string 이라 표시가 없음)."""
    text = {c for c, t in meta["x_dtypes"].items() if t in ("object", "str", "string", "bool", "boolean")}
    return [c for c in meta["features"] if c in set(meta["categorical"]) | text]


# --- split 대회 만들기 ------------------------------------------------------------------------------------------


def default_config(dataset: str) -> dict:
    """r0f0 폴더가 없는 데이터셋의 설정 (FM 맞춤 확인용). 큰 데이터의 tfm_max_rows 는 정하지 않음 (TabArena 처럼 전체)."""
    row = dataset_row(dataset)
    meta = json.loads((fetched(dataset) / "meta.json").read_text())
    task = row["problem_type"]
    cfg = dict(name="", tabarena=dict(dataset=dataset, task_id=int(row["task_id"]), split="r0f0",
                                       categorical=categorical(meta)),
               task=task, target=meta["target"], id_col=ID_COL, metric=METRICS[task],
               cv=dict(n_splits=5, seed=42, stratify=task != "regression"))
    if task != "regression":
        cfg["submit_proba"] = True
    return cfg


def split_comp(dataset: str, split: str) -> Competition:
    """(데이터셋, split) 대회. r0f0 폴더가 있으면 그것 (다른 split 은 그 설정에서 이름·split 만 바꾸고 익명 id 를 뺌),
    없으면 default_config. 생성 폴더에는 config.yaml·features.py 만 (description.md 는 에이전트용이라 r0f0 에만)."""
    name = comp_name(dataset, split)
    if not name.startswith(SPLIT_PREFIX):
        return load_competition(name)
    base = committed().get(dataset)
    if base:
        cfg = yaml.safe_load((base.dir / "config.yaml").read_text())
        cfg["tabarena"] = {k: v for k, v in cfg["tabarena"].items() if k != "anon_id"}
    else:
        cfg = default_config(dataset)
    cfg["name"], cfg["tabarena"]["split"] = name, split
    d = COMP_DIR / name
    d.mkdir(exist_ok=True)
    text = "# 생성 파일 (fab/bench/tabarena/__init__.py 의 split_comp, git 제외). 고치려면 r0f0 설정을 고치고 다시 만듦\n" + \
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if not (d / "config.yaml").exists() or (d / "config.yaml").read_text() != text:
        (d / "config.yaml").write_text(text)
    from fab.bench.tabarena.features import FEATURES_PY

    (d / "features.py").write_text(FEATURES_PY)
    return load_competition(name)


def read_data(dataset: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    d = fetched(dataset)
    meta = json.loads((d / "meta.json").read_text())
    # 범주형은 문자열로 (Amazon 의 id 는 숫자처럼 보이지만 범주). 빈 칸만 결측 (범주 값 "NA" 등을 결측으로 읽지 않게)
    cats = categorical(meta) + ([meta["target"]] if meta["task_type"] == "classification" else [])
    df = pd.read_csv(d / "data.csv", dtype={c: str for c in cats}, keep_default_na=False, na_values=[""])
    return df, meta, pd.read_csv(d / "splits.csv")


def split_rows(splits: pd.DataFrame, n: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) 행 위치. TEST 는 OpenML 분할 그대로, TRAIN 은 같은 repeat 의 나머지 (driver 가 확인)."""
    r, f = parse_split(split)
    test = splits[(splits.repeat == r) & (splits.fold == f)].idx.to_numpy()
    if not len(test):
        raise SystemExit(f"분할에 {split} 없음")
    mask = np.zeros(n, dtype=bool)
    mask[test] = True
    return np.flatnonzero(~mask), np.sort(test)


def prepare(comp: Competition):
    """fab download 가 부름: split 의 public → comp.data_dir, 정답 → ANSWERS/<대회>.{csv,json}.
    행 id = 데이터셋·split 마다 고정 seed 의 무작위 순열 (train·test 가 한 순열을 나눠 가짐 → id 에 정보 없음), 파일은 id 순."""
    ta = comp.tabarena
    df, meta, splits = read_data(ta["dataset"])
    tr, te = split_rows(splits, len(df), ta["split"])
    if ID_COL in df.columns:
        raise SystemExit(f"{ta['dataset']}: 피처에 '{ID_COL}' 열이 있음 → id 열 이름을 바꿔야 함")
    ids = np.random.default_rng([ta["task_id"], fold_index(ta["split"])]).permutation(len(df))
    y = meta["target"]
    feats = [c for c in df.columns if c != y]
    train = df.iloc[tr].assign(**{ID_COL: ids[tr]}).sort_values(ID_COL)[[ID_COL, *feats, y]]
    test = df.iloc[te].assign(**{ID_COL: ids[te]}).sort_values(ID_COL)
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(comp.data_dir / "train.csv", index=False)
    test[[ID_COL, *feats]].to_csv(comp.data_dir / "test.csv", index=False)
    # 클래스 = 학습이 보는 값 (train.csv 를 pandas 로 읽은 타깃의 정렬, fab run 의 LabelEncoder 와 같음)
    y_train = pd.read_csv(comp.data_dir / "train.csv", usecols=[y])[y]
    # CSV 를 pandas 기본값으로 다시 읽으면 'NA'·'None'·'nan' 같은 글자는 결측이 됨 (fab run·에이전트·채점 모두 그렇게 읽음).
    # 피처는 모두에게 같은 결측이라 그대로 둠 (TabArena 데이터의 범주 'nan' 은 원래 결측), 클래스 이름이 그렇게 바뀌면 채점이 틀어지므로 멈춤
    if comp.is_classification and y_train.isna().any():
        bad = sorted(set(df.iloc[tr][y].astype(str)) - set(y_train.dropna().astype(str)))
        raise SystemExit(f"{comp.name}: 클래스 이름 {bad} 가 CSV 에서 결측으로 읽힘 → 클래스 이름을 바꿔야 함")
    classes = sorted(y_train.unique().tolist()) if comp.is_classification else None
    if comp.task == "binary":
        sample = pd.DataFrame({ID_COL: test[ID_COL], y: 0.5})
    elif comp.task == "multiclass":
        sample = pd.concat([test[[ID_COL]].reset_index(drop=True),
                            pd.DataFrame(1 / len(classes), index=range(len(test)), columns=[str(c) for c in classes])],
                           axis=1)
    else:
        sample = pd.DataFrame({ID_COL: test[ID_COL], y: float(y_train.mean())})
    sample.to_csv(comp.data_dir / "sample_submission.csv", index=False)
    if (comp.dir / "description.md").exists():
        shutil.copy(comp.dir / "description.md", comp.data_dir / "description.md")
    ANSWERS.mkdir(parents=True, exist_ok=True)
    test[[ID_COL, y]].to_csv(ANSWERS / f"{comp.name}.csv", index=False)
    info = dict(comp=comp.name, dataset=ta["dataset"], task_id=ta["task_id"], split=ta["split"], task=comp.task,
                metric=comp.metric, target=y, classes=[str(c) for c in classes] if classes else None,
                positive=str(classes[1]) if comp.task == "binary" else None, n_train=len(train), n_test=len(test))
    (ANSWERS / f"{comp.name}.json").write_text(json.dumps(info, indent=1))
    print(f"{comp.name}: train {len(train)} · test {len(test)} → {comp.data_dir} (정답 {ANSWERS / comp.name}.csv)")


def ensure_prepared(comp: Competition):
    """없으면 prepare. 있으면 description.md 만 대회 폴더의 것으로 맞춤 (설명을 고친 뒤 data/ 의 옛 사본이 남지 않게)."""
    if not ((comp.data_dir / "train.csv").exists() and (ANSWERS / f"{comp.name}.json").exists()):
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


def pick_splits(dataset: str, spec: str) -> list[str]:
    allowed = split_names(dataset)
    if spec == "all":
        return allowed
    out = spec.split(",")
    if bad := [s for s in out if s not in allowed]:
        raise SystemExit(f"{dataset} 에 없는 split: {bad} (있는 것: {allowed[0]}~{allowed[-1]})")
    return out


# --- 채점 ------------------------------------------------------------------------------------------------------


def score_submission(sub_csv: Path, answers_csv: Path, info: dict) -> dict:
    """제출 파일 → {score, metric_error, valid_submission} 또는 {valid_submission: False, error}. 에이전트·FM 이 같이 씀.
    metric_error 는 TabArena 와 같은 꼴 (AUC 는 1 − AUC, logloss·rmse 는 그대로). 다중 분류는 확률을 [1e-15, 1] 로 자르고
    행 합을 1 로 맞춤 (Kaggle·MLE-bench 의 logloss 채점과 같음)."""
    from fab.core.metrics import score

    ans = pd.read_csv(answers_csv, dtype={info["target"]: str} if info["task"] != "regression" else None)
    try:
        sub = pd.read_csv(sub_csv)
    except Exception as e:  # noqa: BLE001
        return dict(valid_submission=False, error=f"읽기 실패: {e!r}")
    y = info["target"]
    want = [ID_COL, *(info["classes"] if info["task"] == "multiclass" else [y])]
    if list(map(str, sub.columns)) != want:
        return dict(valid_submission=False, error=f"열 {list(sub.columns)} ≠ {want}")
    sub.columns = want
    if len(sub) != len(ans) or set(sub[ID_COL]) != set(ans[ID_COL]):
        return dict(valid_submission=False, error=f"행·id 가 test 와 다름 ({len(sub)}행, test {len(ans)}행)")
    sub = sub.set_index(ID_COL).loc[ans[ID_COL]]
    try:
        pred = sub.to_numpy(dtype=float)
    except (TypeError, ValueError) as e:  # 확률 대신 레이블 ('Yes'·'No') 을 낸 제출 등
        return dict(valid_submission=False, error=f"예측이 숫자가 아님: {e!r}"[:300])
    if not np.isfinite(pred).all():
        return dict(valid_submission=False, error="예측에 NaN/inf")
    if info["task"] == "binary":
        p = pred[:, 0]
        target, proba = (ans[y] == info["positive"]).astype(int).to_numpy(), np.column_stack([1 - p, p])
    elif info["task"] == "multiclass":
        proba = np.clip(pred, 1e-15, None)
        proba /= proba.sum(1, keepdims=True)
        if bad := set(ans[y]) - set(info["classes"]):
            return dict(valid_submission=False, error=f"정답에 제출 열에 없는 클래스 {sorted(bad)}")
        target = ans[y].map({c: i for i, c in enumerate(info["classes"])}).to_numpy()
    else:
        target, proba = ans[y].to_numpy(dtype=float), pred[:, 0]
    try:
        s = score(info["metric"], target, proba)
    except ValueError as e:  # 예: 확률이 [0, 1] 밖이라 sklearn 이 거부
        return dict(valid_submission=False, error=f"채점 실패: {e!r}"[:300])
    return dict(valid_submission=True, score=s, metric_error=1 - s if info["metric"] == "auc" else s)


def grade_row(comp: Competition, model: str, report: dict, note: str = "") -> dict:
    from fab.core.results import test_info

    ta = comp.tabarena
    return dict(timestamp=pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds"), comp=comp.name,
                dataset=ta["dataset"], task_id=ta["task_id"], split=ta["split"], fold=fold_index(ta["split"]),
                model=model, metric=comp.metric, score=report.get("score"), metric_error=report.get("metric_error"),
                run=(test_info(comp.name, model) or {}).get("run"), note=note)


def grade(comps: list[str], models: list[str], note: str, grade_fn) -> tuple[list[dict], dict[str, str]]:
    """(대회, 모델) 마다 test.npy → 제출 파일 → grade_fn(대회, csv) 로 채점 → results/tabarena.csv.
    원격 정리 작업이 부름 (grade_fn = remote.Grader.grade). 예측이 없거나 채점에 실패한 것은 표에 쓰지 않음."""
    from fab.core.results import test_info
    from fab.submission import make_submission

    rows, errors = [], {}
    for name in comps:
        comp = load_competition(name)
        for m in models:
            if test_info(name, m) is None:
                errors[f"{name}/{m}"] = "test 예측 없음"
                continue
            try:
                report = grade_fn(name, make_submission(name, [m]))
            except Exception as e:  # noqa: BLE001 - 예: test.npy 에 NaN (normalize_proba), classes 불일치
                errors[f"{name}/{m}"] = f"채점 실패: {e!r}"[:500]
                continue
            if not report.get("valid_submission"):
                errors[f"{name}/{m}"] = f"채점 실패: {report.get('error')}"
                continue
            rows.append(grade_row(comp, m, report, note))
            print(f"{name} {m}: {comp.metric}={report['score']:.5f} (metric_error {report['metric_error']:.5f})")
    if rows:
        append_rows(pd.DataFrame(rows), GRADES)
    return rows, errors


def request_grade(comps: list[str], models: list[str], note: str = ""):
    """채점 요청을 Volume 에 → 원격 정리 작업이 요청한 모델의 학습이 끝나기를 기다렸다가 채점 (MLE-bench 와 같은 흐름)."""
    from datetime import datetime

    from fab.bench.tabarena.remote import upload_answers
    from fab.remote import settings, submit_item

    cfg = settings()
    upload_answers(cfg, comps)
    key = f"tabarena__{datetime.now():%Y%m%d-%H%M%S}"
    submit_item(cfg, "grade", key, dict(bench="tabarena", comps=comps, models=models, note=note))
    print(f"채점 요청: {key} (대회 {len(comps)}개 × 모델 {len(models)}개) → 학습이 끝난 뒤 fab remote kick, 결과는 fab sync")


# --- FM·GBDT 제출 ----------------------------------------------------------------------------------------------


def check_models(models: list[str]):
    """제출 전에 모델 이름 확인: fab.models 의 REGISTRY·VARIANTS (컨텍스트 래퍼 @spec 허용). 피처 변형 (~) 은 split 대회에
    없어서 거부."""
    from fab.models import REGISTRY, VARIANTS

    bad = [m for m in models if "~" in m or m.partition("@")[0] not in {*REGISTRY, *VARIANTS}]
    if bad:
        raise SystemExit(f"알 수 없는 모델 (또는 피처 변형 ~): {bad}. 사용 가능: {', '.join([*REGISTRY, *VARIANTS])}")


def submit_fm(dataset: str, splits: list[str], models: list[str], note: str, timeout: int, accelerator: str | None,
              grade_after: bool = True, parts: tuple[str, ...] = ("gpu", "cpu")) -> list[dict]:
    """데이터셋 하나의 split 들을 GPU 작업 하나 (TFM) + CPU 작업 하나 (GBDT) 로 제출 (split 마다 작업을 내면 816개라).
    작업 안에서 split 대회마다 `fab run` 을 차례로 (fab/runner/databricks.py 의 jobs). parts 로 한쪽만 (다시 내기는 gpu 만).
    돌려주는 실행 기록의 cpu 가 True 면 CPU 작업."""
    from fab.bench.tabarena.remote import upload_data
    from fab.models import base_name
    from fab.runner.databricks import CPU_MODELS, record, submit, submit_cpu
    from fab.runner.databricks import settings as dbx_settings

    check_models(models)
    cfg = dbx_settings(argparse.Namespace(accelerator=accelerator, image=None))
    comps = make_comps(dataset, splits)
    data = upload_data(cfg, comps)
    cpu = [m for m in models if base_name(m) in CPU_MODELS]
    gpu = [m for m in models if m not in cpu]
    label = f"ta-{slug(dataset)}-{len(comps)}splits"
    runs = []
    for ms, is_cpu in ((gpu, False), (cpu, True)):
        if not ms or ("cpu" if is_cpu else "gpu") not in parts:
            continue
        jobs = [(c.name, data[c.name], [*ms, "--note", note]) for c in comps]
        if is_cpu:
            run = submit_cpu(cfg, label, None, None, ms, timeout, "", jobs=jobs)
        else:
            run = submit(cfg, label, None, None, ms, cfg["accelerator"], timeout, "", jobs=jobs)
        record(cfg, run)
        runs.append(run)
    if grade_after:
        request_grade([c.name for c in comps], models, note)
    return runs


# 동시 GPU 작업 상한 (#19): 워크스페이스의 GPU 쿼터 (H100 은 8노드, 다른 작업 포함) 를 넘기면 바로 (2초) 'System error
# occurred during execution of task AI Runtime Task: Workspace has exceeded its GPU quota for GPU_1xH100' 로 실패함
# (2026-11-14, docs/tabarena.md 9 의 "운영 기록"). 그렇게 바로 끝난 작업은 기다렸다가 다시 냄
SYSTEM_ERROR = re.compile(r"system error|exceeded its GPU quota", re.I)
FAST_FAIL_SECONDS = 600  # 이보다 짧게 돌고 System error 로 끝난 작업만 다시 냄 (노드가 안 떠 45분 뒤 실패한 것은 안 냄)
RETRY_WAIT = 300  # 다시 내기 전 기다림 (초) × 다시 낸 횟수: 쿼터를 다른 작업이 쓰고 있으면 바로 내도 또 실패


def gpu_state(profile: str, run_id) -> dict:
    """GPU 작업 (AI Runtime) 상태: Jobs API 의 get-run (같은 run_id, fab/remote/reconcile.py 의 run_state)."""
    from fab.runner.databricks import cli_json

    r = cli_json(profile, "jobs", "get-run", str(run_id))
    st = r.get("state") or {}
    start, end = r.get("start_time") or 0, r.get("end_time") or 0
    return dict(life=st.get("life_cycle_state"), result=st.get("result_state"), message=st.get("state_message") or "",
                seconds=(end - start) / 1000 if start and end else None)


def fast_system_error(st: dict) -> bool:
    return st["result"] not in ("SUCCESS", None) and bool(SYSTEM_ERROR.search(st["message"])) \
        and st["seconds"] is not None and st["seconds"] < FAST_FAIL_SECONDS


def submit_fm_throttled(todo: list[tuple[str, list[str]]], models: list[str], note: str, timeout: int,
                        accelerator: str | None, grade_after: bool, max_concurrent: int, retries: int, poll: int = 60):
    """데이터셋 (이름, split) 들을 동시 GPU 작업 max_concurrent 개까지만 내며 차례로 (#19). 로컬 명령이 끝까지 떠 있어야 함
    (낸 작업의 상태를 poll 초마다 보고 자리가 나면 다음을 냄). 0분 System error (fast_system_error) 로 끝난 GPU 작업은
    GPU 쪽만 retries 번까지 다시 냄. 채점 요청은 데이터셋의 GPU 작업이 (다시 내기까지) 끝난 뒤에 냄: 먼저 내면 정리 작업이
    실패한 작업만 보고 '예측 없음' 으로 채점을 끝낼 수 있어서. 세는 것은 이 명령이 낸 GPU 작업뿐 (다른 명령·Agent 작업은 안 셈).
    중간에 멈추면 아직 안 낸 데이터셋과 채점 요청을 못 한 데이터셋을 출력 → fab ta fm·fab ta grade 로 이어서."""
    from fab.models import base_name
    from fab.runner.databricks import CPU_MODELS, settings as dbx_settings

    profile = dbx_settings(argparse.Namespace(accelerator=accelerator, image=None))["profile"]
    has_gpu = any(base_name(m) not in CPU_MODELS for m in models)
    queue = [(ds, splits, 0.0) for ds, splits in todo]  # (데이터셋, split, 이 시각 (monotonic) 전에는 안 냄)
    running: dict[str, dict] = {}  # 데이터셋 → {run (GPU 작업 기록), splits}
    tries: dict[str, int] = {}  # 데이터셋 → 다시 낸 횟수 (처음 낸 것 = 0)
    finished: set[str] = set()  # GPU 작업이 (다시 내기까지) 끝나 채점 요청을 낸 데이터셋
    stamp = lambda: time.strftime("%H:%M:%S")  # noqa: E731
    print(f"동시 GPU 작업 {max_concurrent}개까지, 데이터셋 {len(queue)}개 (System error 다시 내기 {retries}번까지). "
          f"{poll}초마다 상태를 봄 — 끝날 때까지 이 명령을 띄워 둠")
    try:
        while queue or running:
            for ds, r in list(running.items()):
                try:
                    st = gpu_state(profile, r["run"]["run_id"])
                except RuntimeError as e:  # 조회 실패 (CLI 10번 재시도 뒤) → 다음 차례에 다시
                    print(f"{stamp()} {ds}: 상태 조회 실패, 다음에 다시 ({e})")
                    continue
                if st["life"] not in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                    continue
                del running[ds]
                if fast_system_error(st) and tries[ds] < retries:
                    wait_s = RETRY_WAIT * (tries[ds] + 1)
                    print(f"{stamp()} {ds}: {st['seconds']:.0f}초 만에 System error ({st['message'][-150:]}) → "
                          f"{wait_s // 60}분 뒤 GPU 작업 다시 냄 ({tries[ds] + 1}/{retries})")
                    queue.insert(0, (ds, r["splits"], time.monotonic() + wait_s))
                    continue
                took = f"{st['seconds'] / 60:.0f}분" if st["seconds"] is not None else "?"
                print(f"{stamp()} {ds}: GPU 작업 끝 {st['result']} ({took}) {st['message'][:120]}")
                if grade_after:
                    request_grade([comp_name(ds, s) for s in r["splits"]], models, note)
                finished.add(ds)
            while (ready := [i for i, q in enumerate(queue) if q[2] <= time.monotonic()]) \
                    and (len(running) < max_concurrent or not has_gpu):
                ds, splits, _ = queue[ready[0]]
                first = ds not in tries
                runs = submit_fm(ds, splits, models, note, timeout, accelerator, grade_after=False,
                                 parts=("gpu", "cpu") if first else ("gpu",))
                queue.pop(ready[0])  # 제출이 된 뒤에만 (실패해 멈추면 '아직 안 낸' 쪽에 남게)
                tries[ds] = tries.get(ds, -1) + 1
                gpu = [r for r in runs if not r.get("cpu")]
                if gpu:
                    running[ds] = dict(run=gpu[0], splits=splits)
                    print(f"{stamp()} {ds}: GPU 작업 제출 (돌고 있는 것 {len(running)}/{max_concurrent}, 남은 데이터셋 {len(queue)}개)")
                else:  # GBDT 만: 기다릴 GPU 작업 없음
                    if grade_after:
                        request_grade([comp_name(ds, s) for s in splits], models, note)
                    finished.add(ds)
            if queue or running:
                time.sleep(poll)
    except (KeyboardInterrupt, Exception) as e:  # 제출·조회 실패도: 어디까지 냈는지 남기고 멈춤
        left = [q for q, *_ in queue if q not in tries]
        again = [q for q, *_ in queue if q in tries]
        wait = sorted(set(tries) - finished - set(again))
        print(f"\n멈춤 ({e!r}). 낸 작업은 원격에서 계속 돎."
              + (f"\n  아직 안 낸 데이터셋: {' '.join(left)}" if left else "")
              + (f"\n  GPU 작업을 다시 내야 하는 데이터셋 (System error, --models 에 GPU 모델만): {' '.join(again)}"
                 if again else "")
              + (f"\n  채점 요청을 아직 안 한 데이터셋 (작업이 끝난 뒤 fab ta grade <데이터셋...> --models "
                 f"{' '.join(models)}): {' '.join(wait)}" if wait and grade_after else ""))
        raise
    print(f"{stamp()} 모두 끝남")


# --- TabArena 공개 결과 ------------------------------------------------------------------------------------------


def reference() -> pd.DataFrame:
    """공개 결과 parquet (REFERENCE_METHODS) 의 기본 설정 행 → results/tabarena_reference.csv.
    2025-06-12 의 TabPFNv2·TabICL 은 can_run_* 가 False 인 데이터셋 행을 뺌 (RandomForest 대체값)."""
    parts = []
    with tempfile.TemporaryDirectory() as d:
        for method, suite, base in REFERENCE_METHODS:
            out = Path(d) / f"{method}.csv"
            info = _driver("reference", f"{base}/{suite}/methods/{method}/results/model_results.parquet", str(out))
            df = pd.read_csv(out)
            print(f"{method} @ {suite}: {info['rows']}행 {info['methods']}")
            parts.append(df.assign(ta_name=method, ta_suite=suite))
    ref = pd.concat(parts, ignore_index=True)
    ds = datasets()
    if missing := sorted(set(ref.dataset) - set(ds.index)):
        raise SystemExit(f"메타데이터에 없는 데이터셋: {missing}")
    for method, col in CAN_RUN.items():
        drop = (ref.ta_name == method) & ~ref.dataset.map(ds[col]).astype(bool)
        print(f"{method}: {col} False 인 {ref[drop].dataset.nunique()}개 데이터셋 ({drop.sum()}행) 뺌")
        ref = ref[~drop]
    if "imputed" in ref and (imp := ref.imputed.fillna(False).astype(bool)).any():
        print(f"경고: 남은 행 중 대체값 (imputed) {imp.sum()}행: {ref[imp].groupby('ta_name').dataset.nunique().to_dict()}")
    ref.insert(1, "task_id", ref.dataset.map(ds.task_id))
    ref.insert(3, "split", [f"r{f // 3}f{f % 3}" for f in ref.fold])
    ref = ref.sort_values(["ta_name", "task_id", "fold"])
    ref.to_csv(REFERENCE, index=False)
    print(f"{REFERENCE}: {len(ref)}행, 방법 {ref.ta_name.nunique()}개")
    return ref


def default_gbdt(ref: pd.DataFrame, dataset: str, splits: set[str]) -> float:
    """TabArena 기본 GBDT (CatBoost·LightGBM 중 그 split 들의 평균 metric_error 가 나은 쪽) 의 평균. FM 과 같은 split 에서."""
    g = ref[ref.ta_name.isin(["CatBoost", "LightGBM"]) & (ref.dataset == dataset) & ref.split.isin(splits)]
    return float(g.groupby("ta_name").metric_error.mean().min()) if len(g) else np.nan


def compare(models: list[str] | None = None, splits: str = "all") -> pd.DataFrame:
    """우리 FM (results/tabarena.csv) 과 TabArena 의 같은 split 결과 (results/tabarena_reference.csv) 를 데이터셋별로.
    맞춤 기준 (docs/tabarena.md 8, FM 마다): 데이터셋의 80% 이상에서 |우리 split 평균 − TabArena split 평균| ≤ TabArena
    split 표준편차, 그리고 TabArena 기본 GBDT 대비 승패가 같음. 같은 split 이 다 있는 데이터셋만 셈.
    평균·기본 GBDT 는 고른 split (splits) 에서, 표준편차는 늘 데이터셋의 모든 split 에서 (split 사이 흔들림의 크기라서.
    --splits r0f0 처럼 하나만 고르면 그 split 의 표준편차는 0). 같은 (대회, 모델) 을 여러 번 채점했으면 마지막 행."""
    ours = pd.read_csv(GRADES).sort_values("timestamp").drop_duplicates(["comp", "model"], keep="last")
    ref = pd.read_csv(REFERENCE)
    rows = []
    for model in models or sorted(ours.model.unique()):
        ta_name = COUNTERPART.get(model)
        for dataset, o in ours[ours.model == model].groupby("dataset"):
            want = set(split_names(dataset)) if splits == "all" else set(splits.split(","))
            o = o[o.split.isin(want)]
            r_all = ref[(ref.ta_name == ta_name) & (ref.dataset == dataset)] if ta_name else ref.iloc[:0]
            r = r_all[r_all.split.isin(want)]
            full = set(o.split) == want
            row = dict(model=model, ta=ta_name, dataset=dataset, n=len(o), ours=o.metric_error.mean(),
                       ours_std=o.metric_error.std(ddof=0), ta_mean=r.metric_error.mean() if len(r) else np.nan,
                       ta_std=r_all.metric_error.std(ddof=0) if len(r_all) > 1 else np.nan,
                       gbdt=default_gbdt(ref, dataset, want), complete=full and len(r) == len(want))
            row["within"] = abs(row["ours"] - row["ta_mean"]) <= row["ta_std"] if row["complete"] else None
            row["ours_wins"] = row["ours"] < row["gbdt"]
            row["ta_wins"] = row["ta_mean"] < row["gbdt"] if len(r) else None
            row["same_side"] = row["ours_wins"] == row["ta_wins"] if row["complete"] else None
            # 비김 (2026-11-14 정함): TabArena FM 과 기본 GBDT 의 차이가 TabArena split 표준편차 이하면 승패를 세지 않음
            row["decisive"] = abs(row["ta_mean"] - row["gbdt"]) > row["ta_std"] if row["complete"] else None
            rows.append(row)
    return pd.DataFrame(rows)


def summary(cmp: pd.DataFrame) -> pd.DataFrame:
    """FM 마다 맞춤 기준 판정 (docs/tabarena.md 8, 비김 처리는 2026-11-14):
    within 이 데이터셋의 80% 이상 + 승패를 세는 데이터셋 (decisive: |TabArena FM − 기본 GBDT| > TabArena split 표준편차) 에서
    승패가 모두 같음. same_side_all 은 비김까지 센 예전 판정 (참고)."""
    out = []
    for model, g in cmp[cmp.complete.eq(True)].groupby("model"):
        # object 열의 numpy bool 을 그대로 sum 하면 논리합 (True + True = True) → int 로
        within, same = g.within.astype(bool).astype(int), g.same_side.astype(bool).astype(int)
        dec = g.decisive.astype(bool)
        out.append(dict(model=model, ta=g.ta.iloc[0], datasets=len(g), within=int(within.sum()),
                        within_pct=round(100 * within.mean(), 1), decisive=int(dec.sum()),
                        same_side=int(same[dec].sum()), same_side_all=int(same.sum()),
                        passes=bool(within.mean() >= 0.8 and same[dec].all())))
    return pd.DataFrame(out)


# --- 명령 ------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    argv = sys.argv[2:] if argv is None else argv
    if argv[:1] == ["agent"]:
        from fab.agent import main as agent_main

        return agent_main(argv[1:], bench="tabarena")
    ap = argparse.ArgumentParser(prog="fab ta", description="TabArena-v0.1 (docs/tabarena.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="OpenML 데이터·분할 받기 → split 대회 (data/ 에 public, 정답은 FAB_TABARENA)")
    p.add_argument("datasets", nargs="+", help="데이터셋 또는 대회 이름, all = 51개")
    p.add_argument("--splits", default="r0f0", help="all 또는 r0f0,r1f2 처럼 (기본 r0f0)")
    f = sub.add_parser("fm", help="FM·GBDT 제출 (데이터셋마다 GPU 작업 하나 + CPU 작업 하나) + 채점 요청")
    f.add_argument("datasets", nargs="+")
    f.add_argument("--models", nargs="+", required=True)
    f.add_argument("--splits", default="all")
    f.add_argument("--note", default="")
    f.add_argument("--timeout", type=int, default=720, help="작업 시간 제한 (분)")
    f.add_argument("--accelerator", default=None)
    f.add_argument("--no-grade", action="store_true", help="채점 요청을 따로 (fab ta grade)")
    f.add_argument("--max-concurrent", type=int, default=5,
                   help="동시 GPU 작업 상한 (기본 5, H100 은 7개쯤 넘게 내면 바로 실패). 끝날 때까지 이 명령이 떠 있음. "
                        "0 = 상한 없이 한꺼번에 내고 바로 끝남 (예전 방식)")
    f.add_argument("--retries", type=int, default=2, help="0분 System error 로 끝난 GPU 작업을 다시 낼 횟수 (상한 모드에서)")
    f.add_argument("--poll", type=int, default=60, help="상태를 볼 간격 (초, 상한 모드에서)")
    g = sub.add_parser("grade", help="채점 요청 (원격 정리 작업) → fab sync 로 results/tabarena.csv")
    g.add_argument("datasets", nargs="+")
    g.add_argument("--models", nargs="+", required=True)
    g.add_argument("--splits", default="all")
    g.add_argument("--note", default="")
    sub.add_parser("reference", help="TabArena 공개 결과 (기본 설정) → results/tabarena_reference.csv")
    c = sub.add_parser("compare", help="우리 채점과 TabArena 결과 비교 (맞춤 기준)")
    c.add_argument("--models", nargs="*")
    c.add_argument("--splits", default="all", help="all 또는 r0f0 (Lite) 등")
    c.add_argument("--detail", action="store_true", help="데이터셋별 표도")
    sub.add_parser("agent", help="에이전트 실행 (fab ta agent -h, fab mleb agent 와 같은 명령)")
    args = ap.parse_args(argv)

    def names(xs):
        return list(datasets().index) if xs == ["all"] else list(dict.fromkeys(resolve(x) for x in xs))

    if args.cmd == "prepare":
        for ds in names(args.datasets):
            make_comps(ds, pick_splits(ds, args.splits))
    elif args.cmd == "fm":
        check_models(args.models)  # 데이터셋을 하나라도 내기 전에
        todo = [(ds, pick_splits(ds, args.splits)) for ds in names(args.datasets)]  # split 이름도 내기 전에 확인
        if args.max_concurrent > 0:
            submit_fm_throttled(todo, args.models, args.note, args.timeout, args.accelerator, not args.no_grade,
                                args.max_concurrent, args.retries, args.poll)
        else:
            for ds, splits in todo:
                submit_fm(ds, splits, args.models, args.note, args.timeout, args.accelerator, not args.no_grade)
        print("끝난 뒤 fab remote kick → 결과 합치기·채점, fab sync 로 results/tabarena.csv")
    elif args.cmd == "grade":
        check_models(args.models)
        comps = [c.name for ds in names(args.datasets) for c in make_comps(ds, pick_splits(ds, args.splits))]
        request_grade(comps, args.models, args.note)
    elif args.cmd == "reference":
        reference()
    elif args.cmd == "compare":
        cmp = compare(args.models or None, args.splits)
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            if args.detail:
                print(cmp.round(5).to_string(index=False))
            print(summary(cmp).to_string(index=False))
