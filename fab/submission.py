"""저장된 테스트 예측(test.npy)으로 제출 파일 생성. 여러 모델을 주면 예측을 단순 평균.

벤치마크 채점이 부름 (원격 정리 작업 안에서, fab/remote/reconcile.py). 명령은 없음.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import TRANSFORMS, load_competition
from fab.core.results import OUTPUTS, test_info


def make_submission(comp_name: str, models: list[str]) -> Path:
    """모델들의 test.npy 평균으로 제출 파일을 만들어 경로를 반환."""
    comp = load_competition(comp_name)
    # 메모리: test 의 id 열 하나만 읽고, 예측은 모델마다 하나씩 읽어 더함 (모델 수에 비례하지 않음)
    test = pd.read_csv(comp.data_dir / comp.test_file, usecols=[comp.id_col] if comp.id_col else [0])
    pred = None
    for m in models:
        p = np.load(OUTPUTS / comp.name / m / "test.npy")
        if pred is None:
            pred = p.astype(float, copy=True)
        else:
            pred += p
        del p
    pred /= len(models)

    ids = test[comp.id_col] if comp.id_col else np.arange(1, len(test) + 1)
    id_name, target_name = comp.submit_columns or [comp.id_col, comp.target]
    if comp.is_classification:
        # 예측 열 순서: test.json (예측과 함께 저장) → 없으면 meta.json (--no-test 재실행이 덮어썼을 수 있음)
        classes = (test_info(comp.name, models[0]) or {}).get("classes") or \
            json.loads((OUTPUTS / comp.name / models[0] / "meta.json").read_text())["classes"]
    if comp.task == "multiclass" and comp.submit_proba:  # 클래스마다 확률 열 하나 (열 이름 = 클래스 값)
        # 행 합을 1 로 다시 맞춤: TFM 의 float32 확률을 fold·모델 평균하면 합이 1 에서 조금 벗어남
        # (MLE-bench leaf-classification 채점기는 합이 1 ± 1e-6 이 아니면 제출 무효)
        pred = np.clip(pred, 0, None)
        pred /= pred.sum(axis=1, keepdims=True)
        sub = pd.concat([pd.DataFrame({id_name: ids}), pd.DataFrame(pred, columns=classes)], axis=1)
    else:
        if comp.is_classification and comp.submit_proba:
            target = pred[:, 1]
        elif comp.is_classification:
            target = np.asarray(classes, dtype=object)[pred.argmax(1)]
        else:
            target = TRANSFORMS[comp.target_transform][1](pred) if comp.target_transform else pred
        sub = pd.DataFrame({id_name: ids, target_name: target})

    path = OUTPUTS / comp.name / "submissions" / f"{'+'.join(models)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"저장: {path}\n{sub.head()}")
    return path
