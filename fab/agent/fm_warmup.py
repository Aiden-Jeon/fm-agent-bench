"""C2 (+fm) 사전 점검: fm_notes.txt 의 모델별 사용 예를 그대로 작은 합성 데이터로 실행 (가중치 받기 + 예가 실제로 도는지).

    python fm_warmup.py <fm_notes.txt> <모델 이름>     # 성공하면 exit 0, 마지막 줄에 결과 JSON

실행기(runner.py)가 하네스 시작 전에 모델마다 새 프로세스로 부름 → 가중치가 HOME 캐시에 남아 에이전트는 받지 않고 씀,
하네스 시간(1시간)에 안 들어감. 에이전트 작업 디렉터리 밖(실행기 코드 폴더)에 있어 사후 검사의 코드 검사에 안 걸림.
fab 을 import 하지 않음 (runner.py 와 같음). 지시문에 넣을 글 조각도 여기서 나눔 (sections, runner.py 가 import).
"""

import json
import re
import sys
import time
from pathlib import Path

SECTION = re.compile(r"^### (\S+)\s*$", re.M)
CODE = re.compile(r"```python\n(.*?)```", re.S)
MIN_ACC, MAX_RMSE = 0.8, 1.0  # 합성 데이터에서 사용 예가 맞게 돈다고 볼 기준 (check 참고)


def sections(text: str) -> tuple[str, dict[str, str]]:
    """fm_notes.txt → (머리말, {모델 이름: 그 모델의 글 조각}). 조각은 '### 이름' 줄부터 다음 '###' 앞까지."""
    parts = SECTION.split(text)
    head, rest = parts[0], parts[1:]
    return head.rstrip() + "\n", {name: f"### {name}\n{body.strip()}\n" for name, body in zip(rest[::2], rest[1::2])}


def snippet(section: str) -> str:
    m = CODE.search(section)
    if not m:
        raise ValueError("python 코드 블록 없음")
    return m.group(1)


def check(name: str, code: str) -> dict:
    """합성 데이터(3 클래스 분류 + 회귀)로 사용 예를 실행하고 출력 모양을 확인."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(350, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)  # 0·1·2
    y_reg = 3 * X[:, 0] - X[:, 3] ** 2 + rng.normal(scale=0.1, size=len(X))
    env = dict(X_train=X[:300], X_test=X[300:], y_train=y[:300], y_train_reg=y_reg[:300])
    t = time.time()
    exec(compile(code, f"<fm_notes:{name}>", "exec"), env)  # noqa: S102 - 저장소의 고정 글 조각
    cols = [str(c) for c in env["proba"].columns] if hasattr(env["proba"], "columns") else None
    proba, pred = np.asarray(env["proba"], dtype=float), np.asarray(env["pred"], dtype=float)
    if proba.shape != (50, 3) or not np.allclose(proba.sum(1), 1, atol=1e-3):
        raise ValueError(f"proba 모양·합이 이상함: {proba.shape}, 합 {proba.sum(1)[:3]}")
    if pred.shape != (50,) or not np.isfinite(pred).all():
        raise ValueError(f"pred 모양이 이상함: {pred.shape}")
    acc = float((proba.argmax(1) == y[300:]).mean())
    rmse = float(np.sqrt(np.mean((pred - y_reg[300:]) ** 2)))
    # 쉬운 합성 데이터라 맞는 사용 예면 정확도 0.98~1.0, RMSE ≈ 0.1 (표준편차 약 3). 열 순서가 클래스와 어긋나면 모양은
    # 맞아도 정확도가 무너짐 (2026-11-12 nomad C2 의 Kumo 사전 점검 acc 0.0: 확률 열 순서가 실행마다 다름)
    if acc < MIN_ACC or rmse > MAX_RMSE:
        raise ValueError(f"사용 예의 결과가 이상함: acc {acc:.3f} (기준 {MIN_ACC}), rmse {rmse:.3f} (기준 {MAX_RMSE}), "
                         f"proba 열 {cols}")
    return dict(seconds=round(time.time() - t, 1), acc=round(acc, 3), rmse=round(rmse, 3),
                y_reg_std=round(float(y_reg.std()), 3), proba_columns=cols)


def main():
    notes, name = Path(sys.argv[1]), sys.argv[2]
    _, secs = sections(notes.read_text())
    out = check(name, snippet(secs[name]))
    print(json.dumps(dict(model=name, **out)), flush=True)


if __name__ == "__main__":
    main()
