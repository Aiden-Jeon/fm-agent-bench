"""TabPFN v3.5 체크포인트를 로컬에서 받음 (첫 사용 때 브라우저로 Prior Labs 라이선스 동의, TABPFN-3.5 Non-Commercial License).
fab 을 import 하지 않음. tabpfn 이 깔린 옆 가상환경 (FAB_ANOMALY/.venv) 에서 `fab ad weights` 가 `-I` 로 부름.
v3.5 는 분류 · 회귀 head 가 한 파일 (tabpfn 9.2.0 의 ModelSource.get_v3_5 기본 파일) → 그 하나만 받음.
받은 파일 경로를 stdout 마지막 줄 JSON 으로 → fab ad weights 가 Volume 의 weights/tabpfn/ 로 올림 (forecast 트랙의 fab fc weights 와 같은 자리)."""

import json

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import download_model, get_cache_dir

VERSION = ModelVersion("v3.5")
name = TabPFNClassifier.create_default_for_version(VERSION).model_path
name = getattr(name, "name", name)
to = get_cache_dir() / str(name)
if not to.exists():
    res = download_model(to, version=VERSION, which="classifier", model_name=str(name))
    if res != "ok":
        raise SystemExit(f"받기 실패: {res}")
print(json.dumps(dict(path=str(to), name=str(name), bytes=to.stat().st_size)))
