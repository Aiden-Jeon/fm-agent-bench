"""`fab run` 을 다른 곳(GPU 별 작업자 프로세스, Databricks)에서 실행할 때 공유하는 인자."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fab.core.competition import feature_pip, load_competition, load_features
from fab.models import REMOTE_PIP, base_name, split_features

CODE_PATHS = ["fab", "competitions"]  # 원격(fab databricks)에서는 이 위치를 cwd 로 두고 `python -m fab run` 실행


def add_run_args(ap: argparse.ArgumentParser, models_nargs: str = "+"):
    """`fab run` 으로 그대로 전달되는 인자. 새 인자를 추가하면 run_argv 에도 반영."""
    ap.add_argument("comp")
    ap.add_argument("models", nargs=models_nargs)
    ap.add_argument("--max-folds", type=int, default=None, help="빠른 확인용: 앞쪽 n개 fold만 실행")
    ap.add_argument("--note", default="", help="results.csv 에 남길 메모")
    ap.add_argument("--no-test", action="store_true", help="실험용: 테스트 예측 생략 (test.npy 저장 안 함)")
    ap.add_argument("--fold", type=int, default=None,
                    help="fold 하나만 실행 → results/<대회>/<모델>/folds/<k>/ 에 저장 (results.csv 는 전체 fold 를 모아 합칠 때)")
    ap.add_argument("--fold-batch", default=None,
                    help="--fold 실행 묶음 이름. 합칠 때 묶음이 같은 fold 만 합침 (fab databricks --split-folds 가 자동으로 붙임)")


def run_argv(args: argparse.Namespace, models: list[str] | None = None) -> list[str]:
    """add_run_args 로 받은 인자를 `fab run` 명령행으로 되돌림 (대회 이름 제외)."""
    out = [*(args.models if models is None else models), "--note", args.note]
    if args.max_folds:
        out += ["--max-folds", str(args.max_folds)]
    if args.no_test:
        out.append("--no-test")
    if args.fold is not None:
        out += ["--fold", str(args.fold)]
    if args.fold_batch:
        out += ["--fold-batch", args.fold_batch]
    return out


def check_run_args(ap: argparse.ArgumentParser, args: argparse.Namespace):
    """원격에 보내기 전에 확인 (원격에서 실패하면 준비 시간만 버림): 피처 변형(~<variant>) 파일, --fold 범위·--max-folds 충돌."""
    comp = load_competition(args.comp)
    seen = set()
    for m in args.models:
        try:
            variant = split_features(m)[1]
            if variant and variant not in seen:
                seen.add(variant)
                load_features(comp, variant)  # 이름·파일·build 확인 (무거운 import 는 build 안이라 로컬에서 읽힘)
        except (ValueError, FileNotFoundError, AttributeError) as e:
            ap.error(str(e))
    if args.fold is None:
        return
    if args.max_folds:
        ap.error("--fold 와 --max-folds 는 같이 쓸 수 없음")
    n = comp.cv["n_splits"]
    if not 0 <= args.fold < n:
        ap.error(f"--fold 는 0 ~ {n - 1}")


def cpu_count() -> int:
    """이 프로세스가 쓸 수 있는 CPU 수 (affinity, cgroup v2 할당량 반영)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def pip_args(models: list[str], comp: str | None = None) -> list[list[str]]:
    """모델과 피처 변형(~<variant>, comp 가 있을 때)이 원격에서 더 설치할 pip 인자 목록."""
    out = []
    for m in models:
        out += [a for a in REMOTE_PIP.get(base_name(m), []) if a not in out]
    if comp:
        c = load_competition(comp)
        for variant in dict.fromkeys(split_features(m)[1] for m in models):
            out += [a for a in feature_pip(c, variant) if a not in out]
    return out

