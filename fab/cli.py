"""`fab <명령> ...` 진입점. 명령마다 해당 모듈의 main(argv) 로 넘김 (필요한 모듈만 import)."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    # 이름: (모듈, 설명)
    "download": ("fab.core.download", "대회 데이터 준비 (MLE-bench 대회는 MLE-bench 분할의 public 만 data/ 에)"),
    "run": ("fab.runner.run", "FM: 고정 CV fold 로 모델 실행 → OOF/테스트 예측, results/results.csv"),
    "databricks": ("fab.runner.databricks", "FM: Databricks AI Runtime(서버리스 GPU)에서 fab run 실행 후 결과 수집"),
    "mleb": ("fab.bench.mlebench", "MLE-bench 채점·공개 에이전트 점수, fab mleb agent: 에이전트 실행"),
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
