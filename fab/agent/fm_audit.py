"""에이전트 환경의 TFM 사용 기록: 파이썬 프로세스가 TFM 패키지를 실제로 import 하면 (모듈 코드 실행이 끝나면) 한 줄 남김.

실행기(runner.py)가 site-packages 에 `_fab_fm_audit.py` 로 복사하고 `_fab_fm_audit.pth` (`import _fab_fm_audit`) 를 둠
→ 그 파이썬으로 뜨는 모든 프로세스가 읽음. 기록은 환경 변수 FAB_FM_AUDIT (로그 파일 경로) 가 있는 프로세스만: 실행기가
하네스에만 넘김 → 하네스와 그 자식(에이전트가 돌린 스크립트)만 남고, 실행기의 사전 점검(fm_warmup.py)은 안 남음.
사후 검사(fab/agent/__init__.py)도 WATCH 를 import 해서 씀 (FAB_FM_AUDIT 가 없으면 import 해도 아무것도 안 함).

왜 코드 검사(tfm_in_code)만으로 안 되나: 코드에 이름만 있고 안 돌린 경우, 설치 여부만 확인한 경우(importlib.util.find_spec
은 모듈을 실행하지 않음 → 여기 안 남음)를 가르려고.

줄 종류 (시각은 epoch 초):
  import: (pid, ppid, argv 앞부분, cwd, 모듈, TFM 이름) — 그 프로세스에서 처음 import 가 끝났을 때
  alive : HEARTBEAT 초마다, TFM 을 import 한 프로세스만 — 강제 종료(SIGTERM·SIGKILL)나 os._exit (multiprocessing fork
          자식, AIDE 의 코드 실행 프로세스) 는 exit 줄이 없어서 마지막 alive 로 끝난 시각을 어림
  exit  : 정상 종료 (atexit)

동작을 바꾸지 않게: loader 를 다른 클래스로 감싸지 않고 (pkg_resources 처럼 loader 종류로 고르는 코드가 깨짐) 그 spec 의
loader 인스턴스의 exec_module 만 바꿈. 어떤 실패도 그 프로세스를 막지 않음.
"""

import os
import sys

WATCH = {"tabpfn": "tabpfn", "tabicl": "tabicl", "sdm": "kumo", "autogluon": "autogluon", "limix": "limix",
         "exaonetabular": "exaone", "causilo": "causilo", "tabdpt": "tabdpt", "mitra": "mitra"}  # 최상위 모듈 → TFM 이름
HEARTBEAT = 30  # 초


def _install(log: str):
    import atexit
    import json
    import threading
    import time

    seen, started = set(), []

    def write(**row):
        try:
            with open(log, "a") as f:
                f.write(json.dumps(dict(t=round(time.time(), 2), pid=os.getpid(), **row)) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def beat():
        while True:
            time.sleep(HEARTBEAT)
            write(event="alive")

    def imported(name):
        seen.add(name)
        write(event="import", module=name, tfm=WATCH[name], ppid=os.getppid(),
              argv=[str(a)[:200] for a in sys.argv[:3]], cwd=os.getcwd())
        if started != [os.getpid()]:  # 프로세스마다 하나 (fork 자식에는 부모의 스레드가 없음)
            started[:] = [os.getpid()]
            threading.Thread(target=beat, daemon=True, name="fab-fm-audit").start()

    class Finder:
        def find_spec(self, name, path=None, target=None):
            if name not in WATCH or name in seen:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is None:
                    continue
                # 클래스 자체가 loader 인 경우(내장·frozen)는 건드리지 않음: 바꾸면 모든 모듈에 걸림
                inner = None if isinstance(spec.loader, type) else getattr(spec.loader, "exec_module", None)
                if inner is not None:
                    def exec_module(module, _inner=inner, _name=name):
                        _inner(module)
                        if _name not in seen:
                            imported(_name)
                    try:
                        spec.loader.exec_module = exec_module  # 이 find_spec 이 만든 loader 인스턴스에만
                    except (AttributeError, TypeError):
                        pass
                return spec
            return None

    sys.meta_path.insert(0, Finder())
    atexit.register(lambda: seen and write(event="exit", modules=sorted(seen)))


try:
    if os.environ.get("FAB_FM_AUDIT"):
        _install(os.environ["FAB_FM_AUDIT"])
except Exception:  # noqa: BLE001 - 기록 실패로 에이전트의 파이썬을 막지 않음
    pass
