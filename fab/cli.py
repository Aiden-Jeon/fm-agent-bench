"""`fab <명령> ...` 진입점. 명령마다 해당 모듈의 main(argv) 로 넘김 (필요한 모듈만 import)."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    # 이름: (모듈, 설명)
    "download": ("fab.core.download", "대회 데이터 준비 (MLE-bench 대회는 MLE-bench 분할의 public 만 data/ 에)"),
    "databricks": ("fab.runner.databricks", "FM: Databricks 에 제출 (TFM: AI Runtime GPU, GBDT: serverless CPU). 결과는 원격 정리 작업 → fab sync"),
    "run": ("fab.runner.run", "원격 전용 (fab databricks 가 원격에서 부름)"),
    "mleb": ("fab.bench.mlebench", "MLE-bench 채점 요청·공개 에이전트 점수, fab mleb agent: 에이전트 실행"),
    "ta": ("fab.bench.tabarena", "TabArena-v0.1: split 대회 준비·FM 제출·채점 요청·TabArena 결과 비교, fab ta agent: 에이전트 실행"),
    "remote": ("fab.remote", "원격 정리 작업: deploy (만들기·갱신) / status / kick"),
    "sync": ("fab.remote.sync", "원격 결과표 → results/*.csv (예측은 --pull 로 골라서)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    return "사용: fab <명령> [인자...]   (명령별 도움말: fab <명령> -h)\n\n" + "\n".join(lines)


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    if argv[0] not in COMMANDS:
        sys.exit(f"알 수 없는 명령 '{argv[0]}'\n\n{usage()}")
    importlib.import_module(COMMANDS[argv[0]][0]).main(argv[1:])
