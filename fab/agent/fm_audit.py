"""에이전트 환경의 TFM 사용 기록: 파이썬 프로세스가 TFM 패키지를 실제로 import 하면 (모듈 코드 실행이 끝나면) 한 줄 남김.

실행기(runner.py)가 site-packages 에 `_fab_fm_audit.py` 로 복사하고 `_fab_fm_audit.pth` (`import _fab_fm_audit`) 를 둠
→ 그 파이썬으로 뜨는 모든 프로세스가 읽음. 기록은 환경 변수 FAB_FM_AUDIT (로그 파일 경로) 가 있는 프로세스만: 실행기가
하네스에만 넘김 → 하네스와 그 자식(에이전트가 돌린 스크립트)만 남고, 실행기의 사전 점검(fm_warmup.py)은 안 남음.

왜 코드 검사(tfm_in_code)만으로 안 되나: 코드에 이름만 있고 안 돌린 경우, 설치 여부만 확인한 경우(importlib.util.find_spec
은 모듈을 실행하지 않음 → 여기 안 남음)를 가르려고. 한 줄 = (시각, pid, ppid, argv 앞부분, 모듈). 프로세스가 정상 종료하면
exit 줄도 남김 (강제 종료면 없음). 어떤 실패도 그 프로세스를 막지 않음.
"""

import os
import sys

WATCH = {"tabpfn": "tabpfn", "tabicl": "tabicl", "sdm": "kumo", "autogluon": "autogluon", "limix": "limix",
         "exaonetabular": "exaone", "causilo": "causilo", "tabdpt": "tabdpt", "mitra": "mitra"}  # 최상위 모듈 → TFM 이름


def _install(log: str):
    import atexit
    import json
    import time

    seen = set()

    def write(**row):
        try:
            with open(log, "a") as f:
                f.write(json.dumps(dict(t=round(time.time(), 2), pid=os.getpid(), **row)) + "\n")
        except Exception:  # noqa: BLE001
            pass

    class Loader:
        """원래 loader 를 감싸 exec_module 이 끝난 뒤 기록 (import 가 성공했을 때만)."""

        def __init__(self, inner, name):
            self.inner, self.name = inner, name

        def __getattr__(self, attr):
            return getattr(self.inner, attr)

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            if self.name not in seen:
                seen.add(self.name)
                write(event="import", module=self.name, tfm=WATCH[self.name], ppid=os.getppid(),
                      argv=[str(a)[:200] for a in sys.argv[:3]], cwd=os.getcwd())

    class Finder:
        def find_spec(self, name, path=None, target=None):
            if name not in WATCH or name in seen:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        spec.loader = Loader(spec.loader, name)
                    return spec
            return None

    sys.meta_path.insert(0, Finder())
    atexit.register(lambda: seen and write(event="exit", modules=sorted(seen)))


try:
    if os.environ.get("FAB_FM_AUDIT"):
        _install(os.environ["FAB_FM_AUDIT"])
except Exception:  # noqa: BLE001 - 기록 실패로 에이전트의 파이썬을 막지 않음
    pass
