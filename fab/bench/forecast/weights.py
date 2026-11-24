"""TabPFN v3.5 회귀 체크포인트를 로컬에서 받음 (첫 사용 때 브라우저로 Prior Labs 라이선스 수락, 비상업 라이선스).
fab 을 import 하지 않음. TabPFN-TS 가 깔린 옆 가상환경 (기본 FAB_FORECAST/.venv-fm) 에서 `fab fc weights` 가 부름.
받은 파일 경로를 stdout 마지막 줄 JSON 으로 → fab fc weights 가 Volume 의 weights/tabpfn/ 로 올림 (원격은 파일 경로로 읽음)."""

import json

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import download_model, get_cache_dir

VERSION = ModelVersion("v3.5")  # tabpfn-time-series 1.4.0 의 TABPFN_MODEL_VERSION
name = TabPFNRegressor.create_default_for_version(VERSION).model_path
name = getattr(name, "name", name)
to = get_cache_dir() / str(name)
if not to.exists():
    res = download_model(to, version=VERSION, which="regressor", model_name=str(name))
    if res != "ok":
        raise SystemExit(f"받기 실패: {res}")
print(json.dumps(dict(path=str(to), name=str(name), bytes=to.stat().st_size)))
