"""C1 의 F2: 원본 열 192 + 이미지 9열 = 201열. MLE-bench 에이전트 AIDE × Sonnet 풀이에서 옮김 (docs/plans/roadmap.md).

출처: 실행 leaf-classification__aide-sonnet__r1__20261112-083418 (test 0.00385, 에이전트 CV 0.01669) 의 최종 풀이
(workspace/logs/exp/best_solution.py = 노드 34920bd2). 에이전트의 피처는 원본 열 + imgfeat (images/<id>.jpg 를 흑백으로 읽어
128 미만 화소의 통계) 이 전부 → 계산식·순서를 그대로 옮김. 모두 사람이 읽을 수 있는 9열이라 f2s 를 따로 두지 않음.
빼 둔 것: 모델 쪽 (StandardScaler·PCA whiten 은 fold 안에서 fit, LR·PCA-LR·MLP 의 로그 공간 섞기와 OOF 로 고른 가중치·온도).
타깃을 쓰는 피처는 없음. 배정·정답 섞기도 없음.

주의: 이미지는 검은 배경(0)에 흰 잎(255) → `< 128` 은 잎이 아니라 배경. 에이전트는 잎이라고 여겼지만 값을 맞추려고 그대로 둠.
그래서 img_area = 배경 비율 = 1 − 잎 비율, 모멘트·이심률·상관은 배경 화소의 것 (잎 모양은 간접으로만 들어감).
배경의 경계 상자는 늘 이미지 전체 → img_extent 는 img_area 와 늘 같고 (중복 열, 에이전트 그대로 둠), img_aspect·img_log_bbox 는
이미지의 종횡비·log 넓이. 이미지는 잎 경계 상자 + 가장자리 여백 약 30px 이라 이미지 크기 ≈ 잎 크기 (원본 192열에 없는 크기 정보).
동일성 확인 (로컬, 2026-11-13): 이 피처로 에이전트 LR 성분 (StratifiedKFold(5, random_state=0), StandardScaler + LR(C=10))
을 다시 돌리면 OOF logloss 0.052495360956868, 에이전트 기록 0.05249536095684915 (차이 1.8e-14). 만드는 시간 약 4초.

에이전트와 다른 점: imgfeat 의 `except Exception: return zeros(9)` 는 뺌 (이미지가 없으면 조용히 0 이 되지 않고 실패).
128 미만 화소가 5개 미만인 이미지를 0 으로 두는 것은 그대로 (MLE-bench 분할 990장 중 해당 없음).
TFM 피처 상한 없음 (201열, 대회 설정에도 tfm_max_features 없음).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fab.core.competition import load_module

_f0 = load_module(Path(__file__).parents[1] / "features.py", "features_leaf_f0")

PIP = [["pillow==13.0.0"]]  # 원격 CPU 작업 환경에 없을 때만 설치 (이미 있으면 건너뜀)
TFM_MAX_FEATURES = None
IMG_COLS = ["img_area", "img_aspect", "img_log_bbox", "img_ecc", "img_log_m20", "img_log_m02", "img_corr",
            "img_log_edge", "img_extent"]


def imgfeat(path: Path) -> np.ndarray:
    """에이전트 imgfeat 그대로. im = 128 미만 화소 = 배경 (잎은 흰색): 배경 비율, 그 경계 상자 (= 이미지 전체) 의 종횡비·log 넓이,
    배경 화소 2차 모멘트의 이심률·log 분산·상관, log 경계 길이 (배경·잎 경계), 경계 상자 안 배경 비율."""
    from PIL import Image

    im = np.array(Image.open(path).convert("L")) < 128
    h, w = im.shape
    ys, xs = np.nonzero(im)
    if len(ys) < 5:
        return np.zeros(9)
    area = im.sum() / (h * w)
    bh = ys.max() - ys.min() + 1
    bw = xs.max() - xs.min() + 1
    cy, cx = ys.mean(), xs.mean()
    m20 = ((xs - cx) ** 2).mean()
    m02 = ((ys - cy) ** 2).mean()
    m11 = ((xs - cx) * (ys - cy)).mean()
    ecc = np.sqrt((m20 - m02) ** 2 + 4 * m11**2) / (m20 + m02 + 1e-9)
    edge = (im ^ np.roll(im, 1, 0)).sum() + (im ^ np.roll(im, 1, 1)).sum()
    return np.array([area, bh / bw, np.log(bh * bw), ecc, np.log(m20 + 1), np.log(m02 + 1),
                     m11 / (np.sqrt(m20 * m02) + 1e-9), np.log(edge + 1), area * h * w / (bh * bw)])


def build(train: pd.DataFrame, test: pd.DataFrame, data_dir: Path):
    X, X_test, _ = _f0.build(train, test)
    out = []
    for df, ids in ((X, train["id"]), (X_test, test["id"])):
        img = np.array([imgfeat(Path(data_dir) / "images" / f"{i}.jpg") for i in ids])
        out.append(pd.concat([df, pd.DataFrame(img, columns=IMG_COLS, index=df.index)], axis=1))
    return *out, []
