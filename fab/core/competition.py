"""대회 설정, 데이터, CV fold, 피처 모듈 로딩."""

from __future__ import annotations

import importlib.util
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
COMP_DIR = ROOT / "competitions"


@dataclass
class Competition:
    name: str
    task: str
    target: str
    id_col: str | None  # 테스트 파일의 id 열. 없으면(null) 1 부터 매긴 행 번호
    metric: str
    cv: dict
    target_transform: str | None = None  # 회귀 타깃 변환 (예: log1p)
    submit_proba: bool = False  # 분류 대회에서 레이블 대신 확률 제출 (이진: 양성 클래스, 다중: 클래스별 열)
    tfm_max_rows: int | None = None  # TFM 컨텍스트 행 상한 (큰 데이터용), GBDT 에는 적용 안 함
    tfm_max_features: int | None = None  # TFM 피처 상한 (넓은 데이터용), 넘으면 LightGBM 중요도 상위만 사용
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    submit_columns: list[str] | None = None  # 제출 파일 열 이름 [id, 예측]. 기본은 [id_col, target]
    mlebench: str | None = None  # MLE-bench 대회 id. 있으면 fab download 가 MLE-bench 분할(public)을 받고 fab mleb grade 로 채점
    # TabArena split 대회: {dataset, task_id, split (r<r>f<f>), categorical, anon_id (r0f0 에이전트용)}. fab/bench/tabarena
    tabarena: dict | None = None

    @property
    def dir(self) -> Path:
        return COMP_DIR / self.name

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary", "multiclass")


TRANSFORMS = {"log1p": (np.log1p, np.expm1)}  # name: (forward, inverse)


def load_competition(name: str) -> Competition:
    cfg = yaml.safe_load((COMP_DIR / name / "config.yaml").read_text())
    return Competition(**cfg)


def load_data(comp: Competition) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (comp.data_dir / comp.train_file).exists():
        raise FileNotFoundError(
            f"{comp.data_dir}/{comp.train_file} 없음. 먼저 `uv run fab download {comp.name}` 실행"
        )
    return pd.read_csv(comp.data_dir / comp.train_file), pd.read_csv(comp.data_dir / comp.test_file)


VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")  # `_` 로 시작하는 파일은 변형이 쓰는 도우미 (예: nomad 의 _geo.py)


def features_path(comp: Competition, variant: str | None = None) -> Path:
    """기본 피처 competitions/<대회>/features.py, 변형은 competitions/<대회>/features/<variant>.py."""
    if variant is None:
        return comp.dir / "features.py"
    if not VARIANT_NAME.fullmatch(variant):
        raise ValueError(f"피처 변형 이름 '{variant}': 영문·숫자로 시작하고 영문·숫자·_·- 만 (도우미 _*.py·경로는 안 됨)")
    path = comp.dir / "features" / f"{variant}.py"
    if not path.exists():
        have = sorted(p.stem for p in (comp.dir / "features").glob("*.py") if not p.stem.startswith("_"))
        raise FileNotFoundError(f"피처 변형 '{variant}' 없음: {path} (있는 것: {', '.join(have) or '없음'})")
    return path


def load_module(path: Path, name: str) -> ModuleType:
    """파이썬 파일 하나를 모듈로 (competitions/ 는 패키지가 아님). 피처 파일끼리 서로 가져다 쓸 때도 이것."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_features(comp: Competition, variant: str | None = None) -> ModuleType:
    """피처 모듈 로드 (features_path). build(train, test) 를 제공해야 함.
    build 가 data_dir 인자를 받으면 대회 데이터 폴더를 넘김 (표 밖 파일을 읽는 변형용, 예: nomad 의 geometry.xyz)."""
    module = load_module(features_path(comp, variant), f"features_{comp.name}_{variant or 'base'}")
    if not callable(getattr(module, "build", None)):
        raise AttributeError(f"{features_path(comp, variant)} 에 build(train, test) 가 없음")
    return module


class Features(NamedTuple):
    X: pd.DataFrame
    X_test: pd.DataFrame
    cat_cols: list[str]
    max_features: int | None  # TFM 피처 상한: 변형 모듈의 TFM_MAX_FEATURES (None = 상한 없음), 없으면 대회 설정


def build_features(comp: Competition, train: pd.DataFrame, test: pd.DataFrame, variant: str | None = None) -> Features:
    """load_features 의 build 를 부름. 변형 모듈은 TFM_MAX_FEATURES 로 TFM 피처 상한을, PIP 로 원격 설치 패키지를 정할 수 있음."""
    mod = load_features(comp, variant)
    if "data_dir" in inspect.signature(mod.build).parameters:
        X, X_test, cat_cols = mod.build(train, test, data_dir=comp.data_dir)
    else:
        X, X_test, cat_cols = mod.build(train, test)
    return Features(X, X_test, cat_cols, getattr(mod, "TFM_MAX_FEATURES", comp.tfm_max_features))


def feature_pip(comp: Competition, variant: str | None) -> list[list[str]]:
    """피처 변형이 원격에서 더 설치할 pip 인자 목록 (변형 모듈의 PIP). 무거운 import 는 build 안에 두어 로컬에서 읽을 수 있게."""
    return list(getattr(load_features(comp, variant), "PIP", [])) if variant else []


def get_folds(comp: Competition, y: np.ndarray) -> np.ndarray:
    """fold 번호 배열. 모든 모델이 같은 fold를 쓰도록 folds.csv 로 고정 저장."""
    path = comp.dir / "folds.csv"
    if path.exists():
        folds = pd.read_csv(path)["fold"].to_numpy()
        if len(folds) == len(y):
            return folds
        raise ValueError(f"{path} 의 길이({len(folds)})가 train({len(y)})과 다름")

    cv = comp.cv
    splitter_cls = StratifiedKFold if cv.get("stratify") and comp.is_classification else KFold
    splitter = splitter_cls(n_splits=cv["n_splits"], shuffle=True, random_state=cv["seed"])
    folds = np.empty(len(y), dtype=int)
    for i, (_, va) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[va] = i
    pd.DataFrame({"fold": folds}).to_csv(path, index=False)
    return folds
