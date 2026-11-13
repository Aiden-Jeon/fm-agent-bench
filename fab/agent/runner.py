"""MLE-bench 에이전트 1회 실행 (Databricks 서버리스 GPU 작업 안에서, 전용 SP 신원으로 돎).

fab 을 import 하지 않음. 같은 디렉터리의 run.json 이 설정 (fab/agent/__init__.py 가 만듦):
  {key, cid, harness, model, alias, hours, continue, fm, data_tgz, out, harness_versions, gateway, upstream_model,
   traces_table}

흐름 (MLE-bench agents/aide/start.sh 를 따름):
  1. public 데이터 풀기 → HOME/data, 지시문 = instructions.txt(경로만 바꿈) + 추가 안내 + 대회 description.md
  2. 일반 ML 패키지 설치 (TFM 없음), 하네스 설치. fm=true (C2, 설정 이름 +fm) 면 TFM 3개를 더 깔고 지시문의 사용 예를
     그대로 돌려 가중치를 받아 둠 (아래 'C2'). TFM import 기록(fm_audit.py)은 모든 실행에 깖
  3. 로컬 프록시 (127.0.0.1): 하네스의 LLM 요청을 FM API 로 넘김. 모델은 run.json 의 model 로 고정,
     SP 토큰은 요청마다 SDK 에서 받음 (1시간 넘게 돌아도 만료 안 됨), 요청마다 토큰 사용량을 proxy.jsonl 에 기록.
     traces_table 이 있으면 LLM 호출마다 span 을 그 UC 표에 쓰고, 하네스의 OTLP trace 도 토큰을 붙여 같은 표로 넘김.
     gateway=true 면 FM API 대신 Unity Gateway(/ai-gateway/...)로 넘김 (아래 'trace')
  4. 제출 형식 검사 서버 (127.0.0.1:5000/validate): 정답 없이 sample_submission 과 열·행 수·id 만 비교
  5. 하네스 실행: 벽시계 hours 시간, 넘으면 프로세스 그룹째 종료. continue=true 면 시간이 남았는데 스스로 끝날 때
     같은 세션을 이어 감 (아래 '이어 가기')
  6. 회수: submission.csv, 작업 디렉터리 코드, 하네스 기록, pip 목록(전·후), meta.json → Volume 의 out/
"""

import functools
import http.client
import json
import os
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

T0 = time.time()
SRC = Path(sys.argv[0]).resolve().parent
CFG = json.loads((SRC / "run.json").read_text())
HOME = Path("/tmp/mleb_home")  # MLE-bench 의 /home 자리 (data/, submission/, instructions.txt)
WORK = Path("/tmp/mleb_work")  # 하네스 작업 디렉터리 (에이전트 코드가 여기 남음)
OUT = Path("/tmp/mleb_out")
SNAP = Path("/tmp/mleb_snap")  # 진행 중 제출 사본을 만드는 곳 (sync_progress)
PROXY_PORT = 8765
VALIDATE_PORT = 5000
TIME_LIMIT = int(CFG["hours"] * 3600)
FM = bool(CFG.get("fm"))  # C2: 환경에 TFM 을 깔고 지시문에 알림 (설정 이름 +fm)
FM_AUDIT = OUT / "fm_audit.jsonl"  # 하네스와 그 자식 프로세스의 TFM import 기록 (fm_audit.py)

# 일반 ML 패키지 (MLE-bench environment/requirements.txt 중 표 형식에 쓰는 것). TFM(tabpfn·tabicl 등)은 넣지 않음
ML_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "hyperopt",
               "bayesian-optimization", "imbalanced-learn", "scikit-optimize", "matplotlib", "seaborn", "statsmodels",
               "pyarrow", "tqdm", "torch"]
# ---------- C2 (+fm, docs/plans/roadmap.md 의 P5) ----------
# 모델: C1 에서 쓴 TFM 셋. 셋 다 라이선스 동의 없이 받음 (TabPFN 은 v2 만: 패키지 기본값인 최신 버전은 브라우저 동의가 필요해
# 원격에서 실패 → 지시문에 v2 쓰는 법을 적음). 버전은 FM 실행 이미지(docker/air/requirements.txt)와 같게.
# 지시문: 이름 + 그대로 도는 사용 예 (fm_notes.txt). 쓰라고 권하지 않음 ("optional"). 사용 예가 없으면 TabPFN 기본값·Kumo 의
# TableTensor API 에서 막혀 "FM 이 도움이 되나" 대신 "API 를 찾아내나" 를 재게 됨.
# 준비 (하네스 시작 전, 1시간에 안 들어감): 설치 → fm_warmup.py 가 fm_notes.txt 의 사용 예를 합성 데이터로 실행 (가중치가
# HOME 캐시에 남음, 예가 실제로 도는지 확인). 하나라도 실패하면 하네스를 시작하지 않음 (실행마다 받은 모델이 달라지지 않게)
FM_PACKAGES = {"tabpfn_v2": ["tabpfn==9.2.0"], "tabicl": ["tabicl==2.2.0"], "kumo": ["structured-data-models==0.2.1"]}
FM_WARMUP_TIMEOUT = 600  # 초, 모델 하나 (가중치 받기 포함)
APT_TIMEOUT = 600  # 초, apt-get 한 번 (update·install). 최악 합: apt 20분 + 사전 점검 30분 + pip → 작업 여유
#   (fab/agent/__init__.py 의 FM_EXTRA_MINUTES) 안
AIDE_PACKAGES = ["openai", "anthropic", "google-generativeai", "backoff", "black", "coolname", "dataclasses-json",
                 "funcy", "genson", "humanize", "python-igraph", "jsonschema", "omegaconf", "rich", "shutup"]


def step(msg):
    print(f"[agent-runner] {msg} ({time.time() - T0:.0f}s)", flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def pip(*args):
    sh([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def pip_list() -> str:
    return sh([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


@functools.cache
def gpu_info() -> tuple[str, ...] | None:
    """nvidia-smi 의 GPU 줄 ("이름, 메모리") 들. 없거나 실패하면 None. 사양 기록과 지시문이 같이 씀 (한 번만 부름).
    드라이버가 멈추면 nvidia-smi 가 끝나지 않을 수 있어 시간 제한."""
    try:
        res = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", capture_output=True, text=True,
                 timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = tuple(line.strip() for line in res.stdout.splitlines() if line.strip())
    return lines if res.returncode == 0 and lines else None


def _cgroup_cpus() -> float | None:
    # cgroup v2: cpu.max = "<quota> <period>" 또는 "max <period>". v1: cfs_quota_us (-1 = 한도 없음) / cfs_period_us
    if cpu := read_text("/sys/fs/cgroup/cpu.max"):
        quota, period = cpu.split()
        return None if quota == "max" else round(int(quota) / int(period), 2)
    if (q := read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")) and (p := read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        return None if int(q) < 0 else round(int(q) / int(p), 2)
    return None


def _cgroup_mem_gb() -> float | None:
    # v2: memory.max = 바이트 또는 "max". v1: limit_in_bytes (한도가 없으면 아주 큰 값)
    mem = read_text("/sys/fs/cgroup/memory.max") or read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if not mem or mem == "max" or int(mem) >= 2**60:
        return None
    return round(int(mem) / 2**30, 1)


def _meminfo() -> dict:
    meminfo, out = read_text("/proc/meminfo") or "", {}
    for key, name in (("MemTotal", "mem_total_gb"), ("MemAvailable", "mem_available_gb")):
        if m := re.search(rf"^{key}:\s+(\d+) kB", meminfo, re.M):
            out[name] = round(int(m.group(1)) / 2**20, 1)
    return out


def _disk() -> dict:
    out = {}
    for path in ("/tmp", str(Path.home())):
        du = shutil.disk_usage(path)
        out[f"disk_free_gb:{path}"] = round(du.free / 2**30, 1)
        out[f"disk_total_gb:{path}"] = round(du.total / 2**30, 1)
    return out


def node_spec() -> dict:
    """노드 사양 (kaggle-master#32: 큰 대회를 트랙 B 에 넣을지 판단). 컨테이너라 호스트 값과 cgroup 한도를 같이 적음 —
    os.cpu_count()·/proc/meminfo 는 호스트 전체를 보여 줄 수 있음. 실제로 쓸 수 있는 양은 fetch 가 그중 작은 값으로 고름.
    항목마다 따로 잼: 하나가 실패하면 그 항목만 errors 에 남고 나머지는 그대로."""
    spec: dict = {"cpu_count": os.cpu_count()}
    parts = {
        "cpu_affinity": lambda: len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpus": _cgroup_cpus,
        "mem_phys_gb": lambda: round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "meminfo": _meminfo,
        "cgroup_mem_gb": _cgroup_mem_gb,
        "disk": _disk,
        "gpu": lambda: list(gpu_info() or []) or None,
    }
    for name, fn in parts.items():
        try:
            value = fn()
        except Exception as e:  # noqa: BLE001 - 사양 측정은 실행을 막지 않음
            spec.setdefault("errors", {})[name] = repr(e)[:200]
            continue
        if isinstance(value, dict):
            spec.update(value)
        elif value is not None:
            spec[name] = value
    return spec


# ---------- LLM 프록시 ----------

# ---------- trace (선택, run.json 의 traces_table) ----------
# traces_table=<catalog>.<schema>.<prefix>_otel_spans 면 (MLflow experiment 에 UC trace 위치로 묶인 표, README 'trace'):
#   - 프록시가 LLM 호출마다 span 하나를 만들어 /api/2.0/otel/v1/traces 로 그 표에 씀 (SP 토큰, X-Databricks-UC-Table-Name).
#     MLflow 형식 속성 (mlflow.spanType=LLM, spanInputs=프록시가 바꾼 뒤의 요청, spanOutputs=응답, 토큰) → experiment 의
#     Traces 탭에서 보임. 요청·응답은 TRACE_BODY_MAX 자에서 자름 (Codex 는 호출마다 컨텍스트 전체를 보냄)
#   - 실행 하나 = trace 하나 (meta.json 의 trace_id): 끝날 때 root span (AGENT) 을 씀. 입력 = 에이전트가 받은 지시문,
#     출력 = 한 줄 요약 (제출·종료 방식·LLM 호출 수·토큰), 설정·결과 값은 fab.* 속성. LLM span 은 root 아래 (Claude Code 는
#     아래처럼 자기 llm_request span 아래). 하네스가 보낸 traceparent 는 fab.harness_trace_id·fab.harness_parent_span_id 로 남김.
#     작업이 통째로 죽어 finally 가 안 돌면 root 가 없어 UI 에는 안 보이고 표에만 남음
#   - Claude Code 의 OTLP trace (interaction·llm_request·tool 실행 span) 도 프록시의 /otel/v1/traces 로 받아 같은 표로 넘김
#     (ug 의 otel-proxy 와 같은 방식: 하네스 설정에 고정 토큰을 넣으면 1시간 실행 중에 만료됨). 넘길 때 span 을 이 실행의
#     trace 로 옮기고 (부모 없는 interaction 은 root 아래, 원래 trace id 는 fab.harness_trace_id), resource 에 fab.run 을 붙임
#     → UI 에서 root > interaction > llm_request > 프록시 LLM span, interaction > tool.
#     Codex 는 끔 (2026-11-11 시험: 6분에 내부 Rust span 약 2.6만 개·작은 trace 수백 개, 도구·명령 내용은 없음 → 명령은
#     codex.jsonl, 요청·응답은 프록시 LLM span). AIDE 는 OTLP 를 안 내보냄
#   - 전송 기록은 otel.jsonl (proxy.jsonl 과 따로: 사후 검사가 proxy.jsonl 줄 수를 LLM 호출 수로 셈). 표에 쓰기 실패는 실행을 막지 않음
# gateway=true (선택): 같은 요청을 FM API 대신 Unity Gateway 로 보냄 → system.ai_gateway.usage 에 남음 (메타스토어에 Gateway
#   trace 표가 설정돼 있으면 거기에도). 모델은 model service 이름(system.ai.*, run.json 의 upstream_model). 요청마다
#   Databricks-Ai-Gateway-Request-Tags (실행 키·하네스·모델) 와 traceparent 를 붙임. x-databricks-use-coding-agent-mode 헤더는
#   일부러 안 넣음 (Smart Routing·코딩 에이전트 전용 처리가 붙을 수 있어 FM API 실행과 조건이 달라짐)
GATEWAY = bool(CFG.get("gateway"))
TRACES_TABLE = CFG.get("traces_table") or None
UPSTREAM_MODEL = CFG.get("upstream_model") or CFG["model"]  # 프록시가 고정하는 모델 (gateway 면 system.ai.*)
TRACE_ID, ROOT_SPAN_ID = os.urandom(16).hex(), os.urandom(8).hex()  # 이 실행의 trace·root span
HARNESS_OTLP = bool(TRACES_TABLE) and CFG["harness"] == "claude-code"  # 하네스 OTLP 를 받아 이 실행의 trace 로 합침
REQUEST_TAGS = json.dumps({"fab_run": CFG["key"], "fab_harness": CFG["harness"], "fab_model": CFG["model"]})
OTEL_PATH, OTEL_TARGET = "/otel/v1/traces", "/api/2.0/otel/v1/traces"
TRACE_BODY_MAX = 1_000_000  # span 의 요청·응답 글자 수 상한
TRACE_FLUSH_SECS = 15
TRACE_BATCH_BYTES = 4_000_000
TRACE_POST_TIMEOUT = 60  # 초. 프록시 span 전송 한 번
TRACE_CLOSE_SECS = 120  # 끝날 때 남은 span 을 보내는 시간 상한 (회수·작업 시간 여유를 잡아먹지 않게, 넘으면 버리고 기록)
RELAY_TIMEOUT = 20  # 초. 하네스 OTLP 중계: 한 번만, 짧게 (하네스 exporter 가 먼저 포기하고 다시 보내면 span 이 겹침)
# 토큰 셈 (사후 검사 fab/agent/__init__.py 의 posthoc 과 같은 규칙. 실행기는 fab 을 import 못 해서 따로 둠):
# Anthropic 의 input_tokens 는 캐시분을 뺀 값이라 캐시 읽기·쓰기를 더함 (OpenAI 는 캐시 포함해서 셈)
INPUT_TOKEN_KEYS = ("input_tokens", "prompt_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
OUTPUT_TOKEN_KEYS = ("output_tokens", "completion_tokens")

CHAT_PATH = "/openai/v1/chat/completions"  # AIDE 가 쓰는 Chat Completions (도구 호출 변환·content 파트 목록 합치기 대상)
ROUTES = {  # 프록시 경로 → FM API 경로 (gateway 면 Unity Gateway 경로)
    "/anthropic/v1/messages": "/serving-endpoints/anthropic/v1/messages",
    CHAT_PATH: "/serving-endpoints/chat/completions",
    "/openai/v1/responses": "/serving-endpoints/responses",
} if not GATEWAY else {
    "/anthropic/v1/messages": "/ai-gateway/anthropic/v1/messages",  # Anthropic 네이티브 (Claude 전용)
    CHAT_PATH: "/ai-gateway/mlflow/v1/chat/completions",  # 통합 API (AIDE: Claude·GPT 둘 다)
    "/openai/v1/responses": "/ai-gateway/openai/v1/responses",  # OpenAI 네이티브 (GPT 전용)
}


def span_context(headers) -> tuple[str, str, str, dict]:
    """LLM 호출 span 의 (trace id, 기본 부모 = root, 이 span id, 하네스 trace 속성). trace 는 늘 이 실행의 것.
    Claude Code 는 LLM 요청에 traceparent (자기 llm_request span) 를 붙임 → 그 span 이 실제로 넘어오면 그 아래로 옮김
    (Tracer.add_llm), 끝까지 안 오면 root 아래 (시간 종료로 죽음·중계 실패)."""
    tp = next((v for k, v in headers.items() if k.lower() == "traceparent"), "")
    m = re.fullmatch(r"[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}", tp.strip().lower())
    harness = {"harness_trace_id": m.group(1), "harness_parent_span_id": m.group(2)} if m else {}
    return TRACE_ID, ROOT_SPAN_ID, os.urandom(8).hex(), harness


def upstream_headers(ctx) -> dict:
    """LLM 요청에 붙일 헤더 (gateway 일 때만): 요청 태그, 이 호출 span 의 traceparent. 하네스가 보낸 traceparent 는
    forward 에서 뺌: 하네스 span 은 이 실행의 trace 로 옮겨지므로 (rewrite_harness_otlp) 그 trace id 로는 이어지지 않음."""
    if not GATEWAY:
        return {}
    return {"Databricks-Ai-Gateway-Request-Tags": REQUEST_TAGS, "traceparent": f"00-{ctx[0]}-{ctx[2]}-01"}


# OTLP protobuf (ExportTraceServiceRequest) 를 직접 씀: 실행기는 패키지 설치 없이 돌아야 하고 (에이전트 환경에 안 섞이게),
# 필요한 필드가 몇 개뿐. 필드 번호는 opentelemetry-proto trace/v1, common/v1, resource/v1
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field: int, data: bytes) -> bytes:  # length-delimited (문자열·바이트·메시지)
    return _varint(field << 3 | 2) + _varint(len(data)) + data


def _anyvalue(v) -> bytes:
    """AnyValue. dict → kvlist, list → array: UC 표의 attributes(variant) 에 객체·배열로 들어감 (MLflow 가 쓰는 모양)."""
    if v is None:
        return b""
    if isinstance(v, dict):
        return _ld(6, b"".join(_ld(1, _ld(1, str(k).encode()) + _ld(2, _anyvalue(x))) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return _ld(5, b"".join(_ld(1, _anyvalue(x)) for x in v))
    if isinstance(v, bool):
        return _varint(2 << 3) + _varint(int(v))
    if isinstance(v, int):
        return _varint(3 << 3) + _varint(v & (2 ** 64 - 1))
    if isinstance(v, float):
        return _varint(4 << 3 | 1) + struct.pack("<d", v)
    return _ld(1, str(v).encode())


def _attrs(field: int, attrs: dict) -> bytes:
    return b"".join(_ld(field, _ld(1, k.encode()) + _ld(2, _anyvalue(v))) for k, v in attrs.items() if v is not None)


def otlp_span(trace_id, parent_id, span_id, name, start, end, attrs, error=None, kind=1) -> bytes:
    """Span 하나 (시각은 time.time() 초). kind 1=INTERNAL, 3=CLIENT. error 가 있으면 status ERROR."""
    status = _ld(2, error.encode()) + _varint(3 << 3) + _varint(2) if error else _varint(3 << 3) + _varint(1)
    return (_ld(1, bytes.fromhex(trace_id)) + _ld(2, bytes.fromhex(span_id))
            + (_ld(4, bytes.fromhex(parent_id)) if parent_id else b"") + _ld(5, name.encode())
            + _varint(6 << 3) + _varint(kind)
            + _varint(7 << 3 | 1) + struct.pack("<Q", int(start * 1e9))
            + _varint(8 << 3 | 1) + struct.pack("<Q", int(end * 1e9)) + _attrs(9, attrs) + _ld(15, status))


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, i


def _pb_fields(msg: bytes):
    """protobuf 메시지의 필드 → (번호, wire type, 내용 (length-delimited 면 길이 뒤의 바이트, 아니면 None), 원래 바이트)."""
    i = 0
    while i < len(msg):
        start = i
        key, i = _read_varint(msg, i)
        wire, content = key & 7, None
        if wire == 2:
            n, j = _read_varint(msg, i)
            i, content = j + n, msg[j:j + n]
        elif wire == 0:
            _, i = _read_varint(msg, i)
        elif wire in (1, 5):
            i += 8 if wire == 1 else 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > len(msg):
            raise ValueError("truncated protobuf")
        yield key >> 3, wire, content, msg[start:i]


def _rewrite(msg: bytes, field: int, fn) -> bytes:
    """msg 의 length-delimited 필드 field 내용을 fn 으로 바꿈 (나머지 필드는 그대로)."""
    return b"".join(_ld(num, fn(c)) if num == field and wire == 2 else raw for num, wire, c, raw in _pb_fields(msg))


class HarnessSpans:
    """넘긴 하네스 span 의 id 와, 하네스 span 이 부모로 가리키는 id (아직 안 넘어왔을 수 있음) 의 시간 범위.
    시간 종료로 하네스가 죽으면 세션 내내 열린 interaction span 은 안 넘어옴 → 그 자식들의 부모가 표에 없음.
    끝날 때 placeholders() 가 빠진 부모 자리에 span 을 만들어 root 아래 붙임 (UI 트리가 끊기지 않게)."""

    def __init__(self):
        self.lock, self.seen, self.wanted = threading.Lock(), set(), {}

    def record(self, span_id: str, parent_id: str | None, start: int, end: int):
        with self.lock:
            self.seen.add(span_id)
            if parent_id and parent_id != ROOT_SPAN_ID:
                lo, hi = self.wanted.get(parent_id, (start, end))
                self.wanted[parent_id] = (min(lo, start), max(hi, end))

    def has(self, span_id: str) -> bool:
        with self.lock:
            return span_id in self.seen

    def placeholders(self) -> list[bytes]:
        with self.lock:
            missing = {k: v for k, v in self.wanted.items() if k not in self.seen}
        return [otlp_span(TRACE_ID, ROOT_SPAN_ID, sid, f"{CFG['harness']} (span not exported)", lo / 1e9, hi / 1e9,
                          {"fab.placeholder": True, "session.id": CFG["key"]}) for sid, (lo, hi) in missing.items()]


HARNESS = HarnessSpans()


def _harness_span(span: bytes) -> bytes:
    """하네스 span 을 이 실행의 trace 로: trace_id(1) 를 바꾸고, 부모(4)가 없으면 root 아래로. 원래 trace id 는 속성으로."""
    fields = list(_pb_fields(span))
    orig = next((c for num, wire, c, _ in fields if num == 1 and wire == 2), b"")
    parent = next((c for num, wire, c, _ in fields if num == 4 and wire == 2 and c), None)
    sid = next((c for num, wire, c, _ in fields if num == 2 and wire == 2), b"")
    times = {num: struct.unpack("<Q", raw[-8:])[0] for num, wire, _, raw in fields if num in (7, 8) and wire == 1}
    if sid:
        HARNESS.record(sid.hex(), parent.hex() if parent else None, times.get(7, 0), times.get(8, times.get(7, 0)))
    rest = b"".join(raw for num, _, _, raw in fields if num not in (1, 4))
    return (_ld(1, bytes.fromhex(TRACE_ID)) + _ld(4, parent or bytes.fromhex(ROOT_SPAN_ID)) + rest
            + _attrs(9, {"fab.harness_trace_id": orig.hex()}))


def rewrite_harness_otlp(body: bytes) -> bytes:
    """하네스 OTLP (ExportTraceServiceRequest) → resource 에 fab.run 을 더하고, span 을 모두 이 실행의 trace 로 옮김
    (UI 에서 실행 하나가 trace 하나). resource 는 단일 메시지 필드라 ResourceSpans 끝에 하나 더 붙이면 protobuf 가 합침."""
    tag = _ld(1, _attrs(1, {"fab.run": CFG["key"]}))

    def resource_spans(rs: bytes) -> bytes:
        return _rewrite(rs, 2, lambda ss: _rewrite(ss, 2, _harness_span)) + tag

    return _rewrite(body, 1, resource_spans)


def otlp_request(spans: list[bytes]) -> bytes:
    resource = _attrs(1, {"service.name": "fab-mleb-agent", "fab.run": CFG["key"]})
    scope_spans = _ld(1, _ld(1, b"fab.agent.runner")) + b"".join(_ld(2, sp) for sp in spans)
    return _ld(1, _ld(1, resource) + _ld(2, scope_spans))


# MLflow 가 UC 표에 쓰는 모양에 맞춤 (2026-11-11, mlflow 3.18 SDK 로 쓴 span 과 비교): mlflow.spanType 등은 그냥 문자열,
# spanInputs·spanOutputs·tokenUsage 는 객체. JSON 문자열로 보내면 따옴표째 문자열로 남아 MLflow 가 span 종류를 못 읽음
TRACE_LOCATION = TRACES_TABLE.removesuffix("_otel_spans") if TRACES_TABLE else None  # <catalog>.<schema>.<prefix>


def body_value(text: str):
    """요청·응답 본문 → span 값. JSON 이고 TRACE_BODY_MAX 자 안이면 객체, 아니면 (스트리밍 SSE·잘린 본문) 문자열."""
    if len(text) <= TRACE_BODY_MAX:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text[:TRACE_BODY_MAX] + f"... [fab: {len(text) - TRACE_BODY_MAX} chars clipped]"


def mlflow_attrs(span_type: str, trace_id: str, inputs, outputs) -> dict:
    return {"mlflow.spanType": span_type, "mlflow.spanInputs": inputs, "mlflow.spanOutputs": outputs,
            "mlflow.traceRequestId": f"trace:/{TRACE_LOCATION}/{trace_id}"}


def otlp_post(ws, host: str, body: bytes, timeout: float) -> tuple[int | None, bytes, str | None]:
    """OTLP protobuf 를 traces_table 로 한 번 보냄 → (상태, 응답 본문, 오류). 예외는 오류로 돌려줌."""
    conn = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        conn.request("POST", OTEL_TARGET, body, {"Content-Type": "application/x-protobuf",
                                                 "X-Databricks-UC-Table-Name": TRACES_TABLE, **ws.config.authenticate()})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data, data.decode("utf-8", "replace")[:300] if resp.status >= 400 else None
    except Exception as e:  # noqa: BLE001 - trace 전송 실패는 실행을 막지 않음
        return None, b"", repr(e)[:300]
    finally:
        conn.close()


class Tracer:
    """프록시 span 을 모아 TRACE_FLUSH_SECS 마다 traces_table 로 보냄. 실패해도 실행은 계속 (otel.jsonl 에 남김).
    flush 는 flush_lock 으로 한 번에 하나 (끝날 때 close 가 진행 중인 주기 flush 를 기다림)."""

    def __init__(self, ws):
        self.ws, self.host = ws, urllib.parse.urlparse(ws.config.host).netloc
        self.spans, self.pending = [], []  # pending: (하네스 부모 span id, build(parent) → span)
        self.lock, self.flush_lock, self.stop = threading.Lock(), threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def add(self, span: bytes):
        with self.lock:
            self.spans.append(span)

    def add_llm(self, harness_parent: str, build):
        """하네스 span 아래에 둘 LLM span: 그 span 이 넘어온 뒤에 만듦 (flush 때 확인). 끝까지 안 오면 root 아래."""
        with self.lock:
            self.pending.append((harness_parent, build))

    def _loop(self):
        while not self.stop.wait(TRACE_FLUSH_SECS):
            self.flush()

    def flush(self, final: bool = False, deadline: float | None = None):
        with self.flush_lock:
            with self.lock:
                pending, self.pending = self.pending, []
            ready = [build(parent if HARNESS.has(parent) else ROOT_SPAN_ID) for parent, build in pending
                     if final or HARNESS.has(parent)]
            with self.lock:
                self.pending = [x for x in pending if not (final or HARNESS.has(x[0]))] + self.pending
                spans, self.spans = self.spans + ready, []
            batch, size, sent = [], 0, 0
            for sp in spans + [None]:
                if sp is None or (batch and size + len(sp) > TRACE_BATCH_BYTES):
                    if batch:
                        if deadline and time.time() > deadline:
                            self._log(dict(status="dropped", spans=len(spans) - sent, error="close time limit"))
                            return
                        self._post(batch, deadline)
                        sent += len(batch)
                    batch, size = [], 0
                if sp is not None:
                    batch.append(sp)
                    size += len(sp)

    def _post(self, batch: list[bytes], deadline: float | None = None):
        body, t, status, err = otlp_request(batch), time.time(), None, None
        for attempt in range(3):  # 끝날 때는 남은 시간 안에서만 (deadline)
            timeout = min(TRACE_POST_TIMEOUT, max(1.0, deadline - time.time())) if deadline else TRACE_POST_TIMEOUT
            status, _, err = otlp_post(self.ws, self.host, body, timeout)
            if status is not None and status < 500 and status != 429:
                break
            if attempt < 2 and not (deadline and time.time() + 2 * (attempt + 1) > deadline):
                time.sleep(2 * (attempt + 1))
            else:
                break
        self._log(dict(status=status, spans=len(batch), bytes=len(body), secs=round(time.time() - t, 1),
                       **({"error": err} if err else {})))

    def _log(self, row: dict):
        with LOG_LOCK, open(OUT / "otel.jsonl", "a") as f:
            f.write(json.dumps(dict(t=round(time.time() - T0, 1), source="fab", **row)) + "\n")

    def close(self):
        """주기 flush 를 멈추고 (진행 중이면 기다림), 빠진 하네스 부모 자리 span 과 남은 span 을 TRACE_CLOSE_SECS 안에 보냄."""
        deadline = time.time() + TRACE_CLOSE_SECS
        self.stop.set()
        self.thread.join(timeout=TRACE_CLOSE_SECS)
        for sp in HARNESS.placeholders():
            self.add(sp)
        self.flush(final=True, deadline=deadline)


TRACER: Tracer | None = None  # run() 이 만듦 (traces_table 이 있을 때)


def token_totals(rows) -> dict:
    """사용량 dict 들 → {input_tokens, output_tokens, total_tokens} (INPUT_TOKEN_KEYS·OUTPUT_TOKEN_KEYS 규칙)."""
    tok = {"input_tokens": sum(r.get(k, 0) for r in rows for k in INPUT_TOKEN_KEYS),
           "output_tokens": sum(r.get(k, 0) for r in rows for k in OUTPUT_TOKEN_KEYS)}
    tok["total_tokens"] = tok["input_tokens"] + tok["output_tokens"]
    return tok


def trace_llm(ctx, path, t, req, req_size, text, status, asked, usage, extra):
    """LLM 호출 하나를 span 으로 (TRACER 가 있을 때). ctx = span_context(), req_size = 보낸 본문 바이트 수
    (req 를 다시 직렬화해 재지 않음: Codex 는 호출마다 1MB 가까이)."""
    if TRACER is None:
        return
    fmt = "anthropic" if path.startswith("/anthropic") else "openai"
    tokens = token_totals([usage])
    inputs = req if req_size <= TRACE_BODY_MAX else body_value(json.dumps(req, ensure_ascii=False))
    attrs = {**mlflow_attrs("LLM", ctx[0], inputs, body_value(text)),
             "mlflow.llm.model": UPSTREAM_MODEL, "mlflow.message.format": fmt, "mlflow.chat.tokenUsage": tokens,
             "session.id": CFG["key"], "gen_ai.request.model": UPSTREAM_MODEL,
             "gen_ai.usage.input_tokens": tokens["input_tokens"], "gen_ai.usage.output_tokens": tokens["output_tokens"],
             "http.response.status_code": status, "fab.path": path, "fab.asked_model": asked,
             "fab.route": "gateway" if GATEWAY else "fm_api", **{f"fab.{k}": v for k, v in {**ctx[3], **extra}.items()}}
    error, end = (f"HTTP {status}" if not status or status >= 400 else None), time.time()

    def build(parent):
        return otlp_span(ctx[0], parent, ctx[2], f"llm {path}", t, end, attrs, error, kind=3)

    if HARNESS_OTLP and (hp := ctx[3].get("harness_parent_span_id")):
        TRACER.add_llm(hp, build)
    else:
        TRACER.add(build(ctx[1]))


def run_summary(meta: dict) -> tuple[str, dict, dict]:
    """root span 출력: 한 줄 요약 (UI 목록의 Output), 결과 속성, 토큰 합 (proxy.jsonl 에서, 사후 검사와 같은 셈)."""
    calls = []
    for line in (OUT / "proxy.jsonl").read_text().splitlines() if (OUT / "proxy.jsonl").exists() else []:
        try:
            calls.append(json.loads(line))
        except ValueError:
            continue
    tok = token_totals(calls)
    result = {k: meta.get(k) for k in ("format_ok", "format_msg", "timed_out", "harness_seconds", "continues",
                                       "runner_error") if meta.get(k) is not None}
    result |= {"llm_calls": len(calls), "llm_errors": sum(1 for c in calls if (c.get("status") or 0) >= 400),
               "waited_429": round(sum(c.get("waited", 0) for c in calls), 1)}
    sub = (meta.get("format_msg") or "submission ok") if meta.get("format_ok") else \
        f"invalid submission: {meta['format_msg']}" if meta.get("format_msg") else "no submission"
    end = "error" if meta.get("runner_error") else "time limit" if meta.get("timed_out") else "exited"
    text = (f"{sub} | {end} after {meta.get('harness_seconds') or 0:.0f}s | {len(calls)} LLM calls, "
            f"{tok['input_tokens']:,} in / {tok['output_tokens']:,} out tokens")
    return text, result, tok


def trace_root(meta: dict):
    """실행 전체 span (끝날 때 한 번). 입력 = 에이전트가 받은 지시문 (대화 형식), 출력 = 결과 한 줄 요약 (토큰 합 포함).
    설정·결과 값은 fab.* 속성. root 에 mlflow.chat.tokenUsage 는 안 넣음 (LLM span 의 값과 겹쳐 셈).
    trace 단위 mlflow.trace.tokenUsage 는 OTLP 로는 안 채워짐 (resource 속성으로 보내도 trace metadata 에 안 들어감, 2026-11-11)"""
    if TRACER is None:
        return
    keys = ("cid", "harness", "model", "alias", "hours", "continue", "fm", "gateway", "upstream_model")
    prompt = (HOME / "instructions.txt").read_text() if (HOME / "instructions.txt").exists() else ""
    text, result, tok = run_summary(meta)
    attrs = {**mlflow_attrs("AGENT", TRACE_ID, {"messages": [{"role": "user", "content": body_value(prompt)}]}, text),
             "session.id": CFG["key"],
             "fab.run": CFG["key"], **{f"fab.{k}": CFG.get(k) for k in keys if not isinstance(CFG.get(k), (dict, list))},
             **{f"fab.result.{k}": v for k, v in result.items()}}
    TRACER.add(otlp_span(TRACE_ID, None, ROOT_SPAN_ID, f"mleb {CFG['harness']} {CFG['cid']}", T0, time.time(), attrs,
                         meta.get("runner_error")))


USAGE_KEYS = ["input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens"]
LOG_LOCK = threading.Lock()
# 두 모델 다 샘플링 파라미터를 거부 (Sonnet 5.5: temperature·top_p 불가, GPT-6.1 Terra: temperature 는 1 만) → 프록시가 뺌.
# AIDE 는 temperature 를 늘 보냄. MLE-bench 의 o1-preview 설정도 temp=1 (기본값) 로 돌림
DROP_PARAMS = ("temperature", "top_p", "top_k")
# FM API 의 워크스페이스 분당 토큰 한도(429)는 인프라 한계 → 하네스에 넘기지 않고 프록시가 기다렸다 다시 보냄.
# 기다린 시간도 벽시계 1시간에 들어감 (proxy.jsonl 의 waited 로 기록).
# P0 (2026-11-09, 5→60초 간격): 429 받은 요청 대부분이 첫 재시도에 통과 → 짧게 시작해 늘림 (kaggle-master#37). 상한 30초,
# 합 약 330초 (이전 315초와 비슷: 한도가 오래 막혀도 그전보다 일찍 포기하지 않음). ±25% 흔들어 재시도가 한꺼번에 몰리지 않게
RETRY_WAITS = (1, 2, 4, 8, 15, *(30,) * 10)  # 초 (흔들기 전)
RETRY_AFTER_MAX = 60  # 429 의 Retry-After(초)는 이 값에서 자름


def retry_wait(base: float, resp) -> float:
    """429 다음에 기다릴 초. 기본은 base ±25%. Retry-After 가 있으면 그보다 일찍 보내지 않음 (최대 RETRY_AFTER_MAX 로
    자르고 0~25% 늦춰 흔듦). 짧은 Retry-After 로 기본 간격보다 줄이지는 않음: 그러면 15번이 몇 초 만에 끝나 합 약 330초
    보장이 깨짐"""
    wait = base * random.uniform(0.75, 1.25)
    try:
        hinted = float(resp.getheader("Retry-After") or "")
    except ValueError:
        return wait
    if hinted > 0:
        wait = max(wait, min(hinted, RETRY_AFTER_MAX) * random.uniform(1.0, 1.25))
    return wait


# AIDE 의 피드백 단계는 Chat Completions 함수 도구 + 강제 tool_choice 를 씀. FM API 에서 Sonnet 5.5 는 강제 tool_choice 를,
# GPT-6.1 Terra 는 Chat Completions 함수 도구 자체를 거부하고, Sonnet 은 json_schema response_format 도 거부함 (2026-11-09).
# → 두 모델 똑같이: 도구를 빼고 "이 스키마의 JSON 만 답하라" 지시를 붙인 뒤, 답의 JSON 을 함수 호출 응답으로 감싸 돌려줌.
#   JSON 이 깨졌거나 필수 키가 없으면 최대 TOOL_TRIES 번 다시 요청. (원래 AIDE 는 네이티브 함수 호출 → 차이로 기록)
TOOL_TRIES = 3


def tools_as_json_prompt(req: dict) -> tuple[dict, dict] | None:
    """함수 도구가 하나 있는 비스트리밍 Chat Completions 요청 → (도구를 뺀 요청, 함수 정의). 해당 없으면 None."""
    tools = req.get("tools") or []
    if req.get("stream") or len(tools) != 1 or tools[0].get("type") != "function":
        return None
    fn = tools[0]["function"]
    req = {k: v for k, v in req.items() if k not in ("tools", "tool_choice", "parallel_tool_calls")}
    note = (f"Respond by calling the function `{fn['name']}`: reply with ONLY a single JSON object (no prose, no code "
            f"fences) that matches this JSON schema for its arguments.\nDescription: {fn.get('description', '')}\n"
            f"Schema: {json.dumps(fn.get('parameters', {}))}")
    req["messages"] = [*req.get("messages", []), {"role": "user", "content": note}]
    return req, fn


def is_text_part(p) -> bool:
    """content 파트 목록에서 글로 쓰는 파트: type 이 없거나 ""·"text" 이고 text 가 문자열."""
    return isinstance(p, dict) and p.get("type") in (None, "", "text") and isinstance(p.get("text"), str)


def message_text(content) -> str:
    """Chat Completions 의 content: 문자열, 또는 파트 목록 (긴 프롬프트에서 Claude 가 [{type: text, text}, ...] 로 답함)."""
    if isinstance(content, list):
        return "".join(p["text"] for p in content if is_text_part(p))
    return content if isinstance(content, str) else ""


class Flattened(NamedTuple):
    body: bytes  # 하네스에 넘길 본문
    types: list[str]  # 본 파트 type 들 (type 이 없거나 "" 면 "text")
    empty: bool  # 합친 글이 빈 choice 가 있었음 (글 파트가 없었음)
    dropped: str | None  # 버린 파트 (is_text_part 아님) 의 JSON 앞 500자. 원래 본문은 trace 에 남음


def flatten_chat_content(data: bytes) -> Flattened | None:
    """비스트리밍 Chat Completions 응답 본문의 choices[].message.content 가 파트 목록이면 message_text 로 합친 문자열로 바꿈.
    바꿀 게 없거나 JSON 이 아니면 None. AIDE 는 content 를 문자열로 가정함 (#10: +fm 의 긴 프롬프트에서 Sonnet 이 일반 응답도
    파트 목록으로 줘서 코드 추출이 TypeError 로 죽음). 글 파트가 아닌 것은 버림 (글 파트가 없으면 빈 문자열: 빈 답과 같게 AIDE 가 처리)"""
    try:
        resp = json.loads(data)
    except ValueError:
        return None
    types, dropped, changed, empty = set(), [], False, False
    for choice in resp.get("choices") or [] if isinstance(resp, dict) else []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            parts = msg["content"]
            types |= {str(p.get("type") or "text") if isinstance(p, dict) else type(p).__name__ for p in parts}
            dropped += [p for p in parts if not is_text_part(p)]
            msg["content"] = message_text(parts)
            changed, empty = True, empty or not msg["content"]
    if not changed:
        return None
    return Flattened(json.dumps(resp, ensure_ascii=False).encode(), sorted(types), empty,
                     json.dumps(dropped, ensure_ascii=False)[:500] if dropped else None)


def parse_tool_json(content, fn: dict) -> dict | None:
    m = re.search(r"\{.*\}", message_text(content), re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    required = fn.get("parameters", {}).get("required", [])
    return obj if isinstance(obj, dict) and all(k in obj for k in required) else None


def make_proxy(ws):
    host = urllib.parse.urlparse(ws.config.host).netloc

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # 응답 끝에 연결을 닫음 → 스트리밍(SSE)을 길이 없이 그대로 넘김

        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_error(404)
            self.record(dict(path=self.path, status=404))

        def do_POST(self):
            otel = bool(TRACES_TABLE) and self.path.split("?")[0] == OTEL_PATH
            try:
                self.relay_otel() if otel else self.forward()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001 - 처리기가 죽으면 하네스는 연결 끊김만 보고 같은 요청을 끝없이 재시도함
                self.record(dict(path=self.path, status=502, proxy_error=repr(e)[:500]),
                            "otel.jsonl" if otel else "proxy.jsonl")
                try:
                    self.respond(502, json.dumps({"error": {"message": f"fab proxy error: {e!r}"[:500]}}).encode())
                except OSError:
                    pass

        def forward(self):
            t = time.time()
            path = self.path.split("?")[0]
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            target = ROUTES.get(path) or (ROUTES["/anthropic/v1/messages"] + path[len("/anthropic/v1/messages"):]
                                          if path.startswith("/anthropic/v1/messages/") else None)
            if not target:
                self.send_error(404)
                return self.record(dict(path=path, status=404))
            # 본문을 JSON 객체로 읽어 모델을 고정한 뒤 다시 씀. 못 읽으면 (압축·객체 아님) 넘기지 않음: 원래 본문을
            # 그대로 넘기면 하네스가 고른 모델로 갈 수 있음 (고정이 깨짐)
            encoding = self.headers.get("Content-Encoding", "identity").lower()
            try:
                if encoding in ("gzip", "x-gzip", "deflate"):
                    body = zlib.decompress(body, 16 + zlib.MAX_WBITS if "gzip" in encoding else zlib.MAX_WBITS)
                elif encoding != "identity":
                    raise ValueError(f"unsupported Content-Encoding {encoding!r}")
                req = json.loads(body)
                if not isinstance(req, dict):
                    raise ValueError(f"request body is {type(req).__name__}, expected a JSON object")
            except (ValueError, zlib.error) as e:
                return self.reject(400, path, f"fab proxy could not parse the request body: {e}")
            asked = req.get("model")
            # FM API 가 거부할 수 있는 요청 옵션을 확인할 수 있게 기록 (kaggle-master#40): anthropic-beta 값, 도구 수
            sent = {k: v for k, v in (("beta", self.headers.get("anthropic-beta")),
                                      ("n_tools", len(req.get("tools") or []) or None)) if v}
            req["model"] = UPSTREAM_MODEL  # 다른 엔드포인트로 못 가게 고정
            dropped = [k for k in DROP_PARAMS if req.pop(k, None) is not None]
            body = json.dumps(req).encode()
            ctx = span_context(self.headers)
            if path == CHAT_PATH and (conv := tools_as_json_prompt(req)):
                return self.tool_call(target, *conv, asked, dropped, t, ctx)
            # accept-encoding 도 뺌: 넘기면 FM API 가 gzip 으로 답하는데 프록시는 content-encoding 헤더를 지우고
            # 본문만 넘겨서 하네스가 못 읽음 (에러 응답이 깨지고, 사용량·에러 기록도 gzip 바이트가 됨).
            # content-encoding 도 뺌: 본문은 위에서 풀어 다시 쓴 JSON
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "authorization", "x-api-key", "content-length", "connection",
                                            "accept-encoding", "content-encoding", "traceparent", "tracestate")}
            headers |= {"Content-Length": str(len(body)), **upstream_headers(ctx)}
            # 비스트리밍 Chat Completions (AIDE) 는 다 받아서 content 파트 목록을 문자열로 바꿔 넘김 (flatten_chat_content).
            # 스트리밍 Chat Completions 는 쓰는 하네스가 없어 그대로 넘김
            buffered = path == CHAT_PATH and not req.get("stream")
            tail, status, waited, retries, conn, flat, failure, started = b"", None, 0, 0, None, None, None, False
            try:
                conn, resp, retries, waited = self.send(target, body, headers)
                status = resp.status
                passed = [(k, v) for k, v in resp.getheaders()
                          if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding")]
                if buffered:
                    # 다 읽고 바꾼 뒤에 상태 줄을 보냄: 읽다 실패하면 (끊김·시간 초과) 아직 아무것도 안 보냈으니 아래에서
                    # 깨끗한 502 를 줌 → AIDE 가 다시 요청
                    data = resp.read()
                    tail = data[-200_000:]  # 기록·trace 는 FM API 가 준 원래 본문
                    if status == 200 and (flat := flatten_chat_content(data)):
                        data = flat.body
                    started = True
                    self.respond(status, data, passed)
                else:
                    started = True
                    self.send_response(status)
                    for k, v in passed:
                        self.send_header(k, v)
                    self.end_headers()
                    while chunk := resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        tail = (tail + chunk)[-200_000:]
            except (OSError, http.client.HTTPException) as e:
                # 하네스에 아직 아무것도 안 보냈으면 (FM API 연결·읽기 실패) 502 → 하네스가 다시 요청. 보내던 중이면
                # (스트리밍 중 끊김, 하네스가 연결을 닫음) 더 보낼 수 없음 → 기록만
                failure = repr(e)[:500]
                if not started:
                    status = 502
                    try:
                        self.respond(502, json.dumps({"error": {"message": f"fab proxy upstream error: {failure}"}}).encode())
                    except OSError:
                        pass
            finally:
                if conn:
                    conn.close()
            text = tail.decode("utf-8", "replace")
            usage = {k: max(map(int, v)) for k in USAGE_KEYS if (v := re.findall(rf'"{k}"\s*:\s*(\d+)', text))}
            err = text[:500] if status and status >= 400 and text else None
            fl = dict(flattened=flat.types, flattened_empty=flat.empty or None, flattened_dropped=flat.dropped) if flat else {}
            fl = {k: v for k, v in fl.items() if v}
            self.record(dict(path=path, asked=asked, status=status, secs=round(time.time() - t, 1), **usage, **sent,
                             **({"error": err} if err else {}), **({"proxy_error": failure} if failure else {}),
                             **({"dropped": dropped} if dropped else {}), **fl,
                             **({"retries_429": retries, "waited": round(waited, 1)} if retries else {})))
            trace_llm(ctx, path, t, req, len(body), text or failure or "", status, asked, usage,
                      dict(sent, retries_429=retries, waited_429=round(waited, 1), dropped=",".join(dropped) or None,
                           proxy_error=failure, **{k: ",".join(v) if isinstance(v, list) else v for k, v in fl.items()}))

        def respond(self, status: int, data: bytes, headers=(("Content-Type", "application/json"),)):
            """다 만든 본문을 한 번에 보냄 (Content-Length 포함). 비스트리밍 응답·프록시 오류가 같이 씀."""
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def reject(self, status: int, path: str, message: str):
            self.respond(status, json.dumps({"error": {"message": message[:500]}}).encode())
            self.record(dict(path=path, status=status, proxy_error=message[:500]))

        def send(self, target: str, body: bytes, headers: dict):
            """FM API 로 보냄. 429 면 RETRY_WAITS 대로 기다렸다 다시 → (conn, resp, 재시도 수, 기다린 초).
            resp 는 아직 안 읽은 상태 (스트리밍은 부른 쪽이 그대로 넘김), conn 은 부른 쪽이 닫음."""
            waited = retries = 0
            for base in (*RETRY_WAITS, None):
                conn = http.client.HTTPSConnection(host, timeout=600)
                try:
                    conn.request("POST", target, body, {**headers, **ws.config.authenticate()})  # SP 토큰 (SDK 가 갱신)
                    resp = conn.getresponse()
                except BaseException:
                    conn.close()
                    raise
                if resp.status != 429 or base is None:
                    return conn, resp, retries, waited
                resp.read()
                conn.close()
                wait = retry_wait(base, resp)
                time.sleep(wait)
                waited, retries = waited + wait, retries + 1
            raise AssertionError("unreachable")

        def upstream(self, target: str, body: bytes, ctx) -> tuple[int, bytes, int, float]:
            """비스트리밍 요청 (429 재시도 포함) → (상태, 본문, 재시도 수, 기다린 초)."""
            headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
            conn, resp, retries, waited = self.send(target, body, headers | upstream_headers(ctx))
            try:
                return resp.status, resp.read(), retries, waited
            finally:
                conn.close()

        def tool_call(self, target, req, fn, asked, dropped, t, ctx):
            """tools_as_json_prompt 참고. 응답은 OpenAI 함수 호출 형식 (finish_reason=tool_calls)."""
            usage, status, data, obj, tries, retries, waited = {}, None, b"", None, 0, 0, 0
            body = json.dumps(req).encode()
            for tries in range(1, TOOL_TRIES + 1):
                status, data, r, w = self.upstream(target, body, ctx)
                retries, waited = retries + r, waited + w
                if status != 200:
                    break
                resp = json.loads(data)
                for k, v in (resp.get("usage") or {}).items():
                    if isinstance(v, int):
                        usage[k] = usage.get(k, 0) + v
                obj = parse_tool_json(resp["choices"][0]["message"].get("content"), fn)
                if obj is not None:
                    msg = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": f"call_fab_{int(time.time() * 1000)}", "type": "function",
                        "function": {"name": fn["name"], "arguments": json.dumps(obj)}}]}
                    resp["choices"][0].update(message=msg, finish_reason="tool_calls")
                    data = json.dumps(resp).encode()
                    break
            err = data.decode("utf-8", "replace")[:500] if status != 200 or obj is None else None
            if status == 200 and obj is None:
                # TOOL_TRIES 번 다 JSON 을 못 얻음 → 200 + 글 답을 주면 AIDE 가 tool_calls 없다고 assert 로 죽음.
                # 5xx 면 AIDE 백엔드가 backoff 로 다시 요청함
                status = 502
                data = json.dumps({"error": {"message": f"fab proxy: no valid `{fn['name']}` JSON after "
                                                        f"{tries} tries"}}).encode()
            self.respond(status or 502, data)
            self.record(dict(path=self.path, asked=asked, status=status, secs=round(time.time() - t, 1),
                             tool_json=fn["name"], tool_tries=tries, tool_ok=obj is not None,
                             **{k: v for k, v in usage.items() if k in USAGE_KEYS},
                             **({"error": err} if err else {}), **({"dropped": dropped} if dropped else {}),
                             **({"retries_429": retries, "waited": round(waited, 1)} if retries else {})))
            trace_llm(ctx, self.path, t, req, len(body), data.decode("utf-8", "replace"), status, asked, usage,
                      dict(tool_json=fn["name"], tool_tries=tries, tool_ok=obj is not None, retries_429=retries,
                           waited_429=round(waited, 1), dropped=",".join(dropped) or None))

        def relay_otel(self):
            """하네스의 OTLP trace (http/protobuf) → /api/2.0/otel/v1/traces 의 traces_table, SP 토큰을 붙여 그대로 넘김.
            압축 본문은 풀어서 보냄 (Databricks OTLP 수집은 비압축 protobuf 기준). 응답(ExportTraceServiceResponse)도 그대로 돌려줌."""
            t = time.time()
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            encoding = self.headers.get("Content-Encoding", "identity").lower()
            if encoding in ("gzip", "x-gzip", "deflate"):
                body = zlib.decompress(body, 16 + zlib.MAX_WBITS if "gzip" in encoding else zlib.MAX_WBITS)
            try:
                body = rewrite_harness_otlp(body)
            except (ValueError, IndexError):  # protobuf 가 아니거나 깨짐 → 그대로 넘김 (서버가 거부하면 otel.jsonl 에 남음)
                pass
            # 한 번만, RELAY_TIMEOUT 안에 (LLM 의 429 기다림·600초 제한을 쓰지 않음). 실패하면 503 → 하네스 exporter 가 판단
            status, data, err = otlp_post(ws, host, body, RELAY_TIMEOUT)
            self.send_response(status or 503)
            self.send_header("Content-Type", "application/x-protobuf" if status and status < 400 else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            # 200 이어도 일부 span 이 거부될 수 있음 (partial_success, protobuf) → 응답 크기만 남기고 내용은 안 풂
            self.record(dict(source="harness", status=status, bytes=len(body), resp_bytes=len(data),
                             secs=round(time.time() - t, 1), **({"error": err} if err else {})), "otel.jsonl")

        def record(self, row, name="proxy.jsonl"):
            with LOG_LOCK, open(OUT / name, "a") as f:
                f.write(json.dumps(dict(t=round(time.time() - T0, 1), **row)) + "\n")

    return Proxy


# ---------- 제출 형식 검사 (MLE-bench 검증 서버 대신, 정답 없이). 에이전트가 읽으므로 메시지는 영어 ----------

def check_submission(path: Path) -> tuple[bool, str]:
    import pandas as pd

    sample = pd.read_csv(HOME / "data" / "sample_submission.csv")
    try:
        sub = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 - 어떤 파일이든 사유를 돌려줌
        return False, f"Could not read the submission as CSV: {e!r}"
    if list(sub.columns) != list(sample.columns):
        return False, f"Columns {list(sub.columns)} do not match the expected columns {list(sample.columns)}"
    if len(sub) != len(sample):
        return False, f"Submission has {len(sub)} rows, expected {len(sample)}"
    id_col = sample.columns[0]
    if set(sub[id_col].astype(str)) != set(sample[id_col].astype(str)):
        return False, f"The {id_col} values do not match the test set ids"
    if sub.drop(columns=id_col).isna().any().any():
        return False, "Submission contains missing predictions"
    return True, "Submission is valid."


class Validate(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        from email.parser import BytesParser  # multipart/form-data (curl -F "file=@...")

        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        msg = BytesParser().parsebytes(f"Content-Type: {self.headers.get('Content-Type', '')}\r\n\r\n".encode() + raw)
        parts = [p for p in msg.walk() if p.get_param("name", header="content-disposition") == "file"]
        tmp = OUT / "validate_upload.csv"
        tmp.write_bytes(parts[0].get_payload(decode=True) or b"" if parts else b"")
        ok, msg = check_submission(tmp)
        with LOG_LOCK, open(OUT / "validate.jsonl", "a") as f:
            f.write(json.dumps(dict(t=round(time.time() - T0, 1), ok=ok, msg=msg)) + "\n")
        body = json.dumps({"result": msg, "is_valid": ok}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(handler, port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


# ---------- 지시문 ----------

def instructions(fm_text: str = "") -> str:
    # GPU 이름만 (P0 의 --query-gpu=name 출력과 같은 글: 지시문이 실행마다 같도록)
    hardware = "\n".join(line.split(",")[0].strip() for line in gpu_info() or ())
    h, m = divmod(TIME_LIMIT // 60, 60)
    # AIDE 는 자기 작업 공간에서 돌아 MLE-bench start.sh 처럼 /home/ 을 지워 상대 경로로 (submission/submission.csv)
    text = (SRC / "instructions.txt").read_text().replace("/home/", "" if CFG["harness"] == "aide" else f"{HOME}/")
    notes = (SRC / "additional_notes.txt").read_text().format(
        hardware=f"1 {hardware}" if hardware else "a CPU", time_limit=f"{h}hrs {m}mins 0secs")
    desc = (HOME / "data" / "description.md").read_text()
    fm = f"{fm_text.strip()}\n\n" if fm_text else ""  # C2: 추가 안내 뒤, 대회 설명 앞
    return f"{text}\n\n{notes}\n\n{fm}COMPETITION INSTRUCTIONS\n------\n\n{desc}"


# ---------- C2: TFM 준비, TFM import 기록 ----------

def build_tools() -> str:
    """Kumo 는 triton 이 커널 런처를 컴파일 (gcc, Python.h). FM 실행 이미지의 베이스(dcs-base)에는 없어서 이미지에 넣었음
    (docs/known-issues.md) → 에이전트 노드에서도 없으면 apt 로. 결과 글은 meta 의 fm_build_tools (실패해도 사전 점검이 판단)."""
    import sysconfig

    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if shutil.which("gcc") and header.exists():
        return "present"
    apt = ["apt-get"] if os.geteuid() == 0 else ["sudo", "-n", "apt-get"] if shutil.which("sudo") else None
    if apt is None:
        return f"missing (gcc={bool(shutil.which('gcc'))}, {header}={header.exists()}), no root"
    pkgs = ["gcc", "libc6-dev", f"python{sys.version_info.major}.{sys.version_info.minor}-dev"]
    try:
        for args in (["update", "-q"], ["install", "-y", "-q", "--no-install-recommends", *pkgs]):
            res = sh([*apt, *args], capture_output=True, text=True, timeout=APT_TIMEOUT,
                     env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
            if res.returncode:
                return f"apt-get {args[0]} failed ({res.returncode}): {(res.stderr or res.stdout)[-300:]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"apt-get failed: {e!r}"[:400]
    return f"installed {' '.join(pkgs)} (gcc={bool(shutil.which('gcc'))}, Python.h={header.exists()})"


def warmup_result(stdout: str, name: str) -> dict:
    """사전 점검 출력에서 결과 줄 ({"model": name, ...} JSON). 라이브러리가 뒤에 다른 줄을 찍어도 찾음."""
    for line in reversed(stdout.splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("model") == name:
            return {k: v for k, v in row.items() if k not in ("ok", "total_seconds", "error")}
    return {"error": f"no result line: {stdout[-500:]}"}


def prepare_fm(meta: dict) -> str:
    """TFM 설치 + 사용 예 실행 (가중치 받기) → 지시문에 넣을 글. 결과는 meta["fm_warmup"] = {모델: {ok, seconds, ...}}
    (meta["fm"] 은 run.json 의 설정값 true 그대로: 여기서 실패해도 사후 검사가 dict 로 잘못 읽지 않게 키를 나눔).
    사전 점검은 모델마다 새 프로세스 (FAB_FM_AUDIT 없이 → import 기록에 안 남음, GPU 메모리도 프로세스와 함께 풀림)."""
    import fm_warmup  # 실행기 코드 폴더 (SRC) 에 같이 올라옴

    for args in FM_PACKAGES.values():
        pip(*args)
    meta["fm_build_tools"] = build_tools()
    step(f"TFM 설치 ({', '.join(FM_PACKAGES)}), 컴파일 도구: {meta['fm_build_tools']}")
    head, secs = fm_warmup.sections((SRC / "fm_notes.txt").read_text())
    env = {k: v for k, v in os.environ.items() if k != "FAB_FM_AUDIT"}
    meta["fm_warmup"] = {}
    for name in FM_PACKAGES:
        t = time.time()
        try:
            res = sh([sys.executable, "-u", str(SRC / "fm_warmup.py"), str(SRC / "fm_notes.txt"), name],
                     capture_output=True, text=True, timeout=FM_WARMUP_TIMEOUT, cwd=SRC, env=env)
            info = warmup_result(res.stdout, name) if res.returncode == 0 else \
                {"error": (res.stderr or res.stdout)[-3000:]}
        except subprocess.TimeoutExpired:
            info = {"error": f"timeout {FM_WARMUP_TIMEOUT}s"}
        meta["fm_warmup"][name] = dict(ok="error" not in info, total_seconds=round(time.time() - t, 1), **info)
        step(f"TFM 사전 점검 {name}: {'ok' if 'error' not in info else 'FAILED'} "
             f"{json.dumps(info, ensure_ascii=False)[-600:]}")
    if failed := [n for n, v in meta["fm_warmup"].items() if not v["ok"]]:
        raise RuntimeError(f"TFM 사전 점검 실패 {failed} → 하네스를 시작하지 않음 (meta.json 의 fm_warmup)")
    return head + "\n" + "\n".join(secs[n] for n in FM_PACKAGES)


def install_audit(meta: dict):
    """fm_audit.py 를 이 파이썬의 site-packages 에 + .pth 로 시작할 때 읽게 (모든 실행, 에이전트에게는 안 알림).
    기록은 FAB_FM_AUDIT 가 있는 프로세스(하네스와 자식)만. 실패해도 실행은 계속 (meta 의 fm_audit)."""
    import sysconfig

    try:
        site = Path(sysconfig.get_paths()["purelib"])
        shutil.copy(SRC / "fm_audit.py", site / "_fab_fm_audit.py")
        (site / "_fab_fm_audit.pth").write_text("import _fab_fm_audit\n")
        meta["fm_audit"] = str(site)
    except OSError as e:
        meta["fm_audit"] = f"error: {e!r}"[:300]


# ---------- 하네스 ----------

def install_harness():
    h, v = CFG["harness"], CFG["harness_versions"]
    if h == "aide":
        pip(*AIDE_PACKAGES)
        pip("--no-deps", f"git+https://github.com/thesofakillers/aideml@{v['aide']}")
    elif h == "claude-code":
        sh(f"curl -fsSL https://claude.ai/install.sh | bash -s {v['claude-code']}", check=True)
    elif h == "codex":
        if shutil.which("npm"):
            sh(["npm", "install", "-g", f"@openai/codex@{v['codex']}"], check=True)
        else:  # node 가 없으면 GitHub 릴리스의 리눅스 바이너리
            url = (f"https://github.com/openai/codex/releases/download/rust-v{v['codex']}/"
                   "codex-x86_64-unknown-linux-musl.tar.gz")
            dst = Path.home() / ".local" / "bin"
            dst.mkdir(parents=True, exist_ok=True)
            sh(f"curl -fsSL {url} | tar xz -C {dst} && mv {dst}/codex-x86_64-unknown-linux-musl {dst}/codex", check=True)
    else:
        raise SystemExit(f"모르는 하네스: {h}")


def harness_command(prompt_file: Path) -> tuple[list[str], dict, Path]:
    """(명령, 추가 환경 변수, 하네스 출력 파일)."""
    h, model, alias = CFG["harness"], CFG["model"], CFG["alias"]
    proxy = f"http://127.0.0.1:{PROXY_PORT}"
    env = {"PATH": f"{Path.home() / '.local' / 'bin'}:{os.environ['PATH']}",
           "FAB_FM_AUDIT": str(FM_AUDIT)}  # 하네스와 자식 프로세스만 TFM import 를 기록 (fm_audit.py)
    prompt = prompt_file.read_text()
    if h == "aide":
        # AIDE 는 모델 이름이 gpt- 로 시작해야 OpenAI 백엔드(=OPENAI_BASE_URL)를 씀 → 별칭, 실제 모델은 프록시가 고정
        env |= {"OPENAI_BASE_URL": f"{proxy}/openai/v1", "OPENAI_API_KEY": "local-proxy"}
        cmd = ["aide", f"data_dir={HOME / 'data'}", f"desc_file={prompt_file}", "exp_name=exp",
               f"log_dir={WORK / 'logs'}", f"workspace_dir={WORK / 'workspaces'}",
               f"agent.code.model={alias}", f"agent.feedback.model={alias}", "agent.steps=500",
               f"agent.time_limit={TIME_LIMIT}", f"exec.timeout={TIME_LIMIT}", "copy_data=False",
               "agent.search.max_debug_depth=20", "agent.search.debug_prob=1",
               # FM API 의 Claude 는 시스템 메시지만 있는 요청을 거부 ('messages' must not be empty).
               # MLE-bench 도 o1-preview 에서 이 옵션을 씀. 두 AIDE 설정(Sonnet·GPT)에 똑같이 켬
               "agent.convert_system_to_user=True"]
    elif h == "claude-code":
        env |= {"ANTHROPIC_BASE_URL": f"{proxy}/anthropic", "ANTHROPIC_AUTH_TOKEN": "local-proxy",
                "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1", "IS_SANDBOX": "1",
                # 도구 검색(일부 도구를 나중에 불러옴)을 끔 (kaggle-master#40): 첫 요청에 붙는 anthropic-beta
                # advanced-tool-use-2025-11-20 을 FM API 가 400 으로 거부함. 프록시에서 헤더만 빼면 본문의 지연 로드 도구가
                # 검색 없이 남을 수 있어서, 기능 자체를 끄고 처음부터 모든 도구를 넘김
                "ENABLE_TOOL_SEARCH": "false"}
        if TRACES_TABLE:  # ug 가 Claude Code 에 넣는 trace 설정 (unity-gateway agents/claude.py), 받는 곳만 로컬 프록시.
            # 토큰은 프록시가 붙임 → otelHeadersHelper 불필요. PROPAGATE_TRACEPARENT: LLM 요청에 traceparent (자기
            # llm_request span) 를 붙임 → 프록시 LLM span 을 그 아래에 둠 (span_context·Tracer.add_llm)
            env |= {"CLAUDE_CODE_ENABLE_TELEMETRY": "1", "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
                    "OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"{proxy}{OTEL_PATH}",
                    "CLAUDE_CODE_PROPAGATE_TRACEPARENT": "1"}
        cmd = ["claude", "-p", prompt, *claude_flags()]
    else:  # codex
        codex_home = WORK / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "config.toml").write_text(
            f'model = "{model}"\nmodel_provider = "fab"\n\n[model_providers.fab]\n'
            f'name = "Databricks FM API (fab proxy)"\nbase_url = "{proxy}/openai/v1"\n'
            f'env_key = "FAB_PROXY_KEY"\nwire_api = "responses"\n')
        # Codex 의 OTLP trace 는 안 켬 (위 'trace': 내부 span 만 많고 도구·명령 내용 없음). 켜려면 ug 처럼
        # [otel.trace_exporter.otlp-http] endpoint = "{proxy}{OTEL_PATH}", protocol = "binary"
        env |= {"CODEX_HOME": str(codex_home), "FAB_PROXY_KEY": "local-proxy"}
        cmd = ["codex", "exec", *CODEX_FLAGS, prompt]
    return cmd, env, OUT / f"{h}.jsonl"


def claude_flags() -> list[str]:
    return ["--model", CFG["model"], "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]


CODEX_FLAGS = ["--json", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"]

# ---------- 이어 가기 (continue 규칙, 2026-11-10 결정, kaggle-master#35) ----------
# 기본(공식 설정)은 하네스가 스스로 끝나면 그대로 끝. run.json 의 continue=true 인 별도 설정(+cont)에서만:
# 하네스가 정상 종료(exit 0)했는데 시간이 CONTINUE_MIN_LEFT 이상 남았으면 같은 세션에 CONTINUE_MESSAGE 를 보내 이어 감.
# 세 하네스에 같은 규칙. AIDE 는 자기 time_limit 까지 돌아 일찍 끝나지 않고 이어 갈 세션도 없어서 해당 없음.
# 상한 (2026-11-11 결정, P0 +cont 2회의 구간별 기록으로): 도구를 하나도 안 쓰고 끝난 구간(무작업)이 연속
# CONTINUE_IDLE_MAX 번이면 멈춤. 무작업 한 번 뒤 다시 작업하는 경우가 있어 1 은 너무 짧고 (nomad 개선 3번을 놓침),
# 2·3·5 는 두 실행 모두 최종 제출이 같았음 → 여유를 하나 둔 3 (5 는 거절 뒤 호출만 늘림). "제출이 안 바뀜" 은 실험 뒤 일부러
# 안 바꾼 구간까지 세서 쓰지 않음. 근거 표는 docs/mle-bench.md 의 "+cont" 절.
# CONTINUE_MAX 는 매번 사소한 도구 호출을 하나씩 해서 무작업으로 안 잡히는 경우를 막는 안전장치.
CONTINUE_MIN_LEFT = 120  # 초
CONTINUE_IDLE_MAX = 3
CONTINUE_MAX = 40
CODEX_NON_TOOL_ITEMS = {"reasoning", "agent_message", "error", "todo_list"}  # 이 밖의 item 은 도구 사용으로 셈
CONTINUE_MESSAGE = (
    "The time limit has not been reached yet: you still have {left} remaining. Keep working on the task and use the "
    "remaining time to improve your submission's score. Keep a valid submission saved at {submission} at all times; "
    "when the time limit is reached your process is terminated.")


SESSION_KEYS = {"claude-code": "session_id", "codex": "thread_id"}  # stream-json / --json 출력에서 세션 id 키


def session_id(log: Path) -> str | None:
    """하네스 출력(jsonl)의 첫 세션 id. 이어 가도 같은 세션이라 id 가 안 바뀜 (2026-11-10 원격 확인: Claude Code 21구간 모두
    같은 session_id) → 한 번 찾아 run_harness 가 기억함. 출력을 처음부터 읽다 첫 id 에서 멈춤."""
    key = SESSION_KEYS.get(CFG["harness"])
    if not key or not log.exists():
        return None
    with open(log, errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and isinstance(ev.get(key), str):
                return ev[key]
    return None


def tool_calls(log: Path, offset: int) -> int | None:
    """하네스 출력(jsonl)의 offset 바이트부터 (= 구간 하나) 도구 호출 수. 셀 수 없는 하네스면 None.
    Claude Code: assistant 메시지의 tool_use 블록. Codex: 생각·답·오류·할 일 목록이 아닌 item (명령 실행·파일 수정 등,
    item.started/completed 가 같은 id 로 둘 다 나오니 id 로 셈, id 가 없는 item 은 completed 만 하나씩.
    새 item 종류는 도구로 쳐서 일찍 멈추지 않는 쪽으로)."""
    if CFG["harness"] not in SESSION_KEYS or not log.exists():
        return None
    n, items = 0, set()
    with open(log, "rb") as f:
        f.seek(offset)
        for k, line in enumerate(f):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if CFG["harness"] == "claude-code" and ev.get("type") == "assistant":
                content = (ev.get("message") or {}).get("content")
                n += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use") \
                    if isinstance(content, list) else 0
            elif CFG["harness"] == "codex" and ev.get("type") in ("item.started", "item.completed"):
                item = ev.get("item")
                if isinstance(item, dict) and item.get("type") not in CODEX_NON_TOOL_ITEMS:
                    if item.get("id") is not None:
                        items.add(item["id"])
                    elif ev["type"] == "item.completed":  # id 가 없으면 started·completed 를 짝지을 수 없음 → 끝난 것만
                        items.add(("line", k))
    return n + len(items)


def continue_command(sid: str | None, left: float) -> list[str] | None:
    """같은 세션을 이어 가는 명령. 이어 갈 수 없으면 None (AIDE, 또는 Codex 인데 세션 id 를 못 찾음)."""
    m, s = divmod(int(left), 60)
    msg = CONTINUE_MESSAGE.format(left=f"{m} minutes {s} seconds", submission=HOME / "submission" / "submission.csv")
    if CFG["harness"] == "claude-code":  # id 가 없으면 --continue (cwd 의 가장 최근 대화, 플래그라 모호하지 않음)
        return ["claude", "-p", msg, *(["--resume", sid] if sid else ["--continue"]), *claude_flags()]
    if CFG["harness"] == "codex" and sid:
        # `exec resume [SESSION_ID] [PROMPT]`: id 없이 --last 를 쓰면 글이 SESSION_ID 자리로 읽힐 수 있음 → id 가 있을 때만.
        # --json·--skip-git-repo-check 등은 resume 하위 명령의 옵션 (codex 0.167.0 `exec resume --help`, 로컬에서 JSON 출력 확인)
        return ["codex", "exec", "resume", *CODEX_FLAGS, sid, msg]
    return None


def run_harness(cmd: list[str], env: dict, log: Path, meta: dict):
    """벽시계 TIME_LIMIT 동안 하네스 실행 (넘으면 프로세스 그룹째 종료). continue 설정이면 스스로 끝날 때 이어 감.
    구간마다 도구 호출 수를 meta["segments"][i]["tools"] 에, 이어 가기를 멈춘 까닭은 meta["continue_stop"] 에 남김."""
    t = time.time()
    meta["harness_started_at"] = round(t, 2)  # epoch 초: fm_audit.jsonl 의 시각과 맞춰 보려고
    deadline = t + TIME_LIMIT
    stop = threading.Event()
    threading.Thread(target=sync_progress, args=(stop, meta), daemon=True).start()
    cont, sid, meta["segments"], idle = bool(CFG.get("continue")), None, [], 0
    # 이어 가기 구간도 같은 출력 파일에 덧붙임 → 사후 검사·토큰 집계가 전체를 봄
    with open(log, "w") as out, open(OUT / "harness_stderr.txt", "w") as err:
        while True:
            seg_t, seg_off = time.time(), log.stat().st_size  # 구간 시작 바이트 (하네스가 이 파일 끝에 이어 씀)
            proc = subprocess.Popen(cmd, cwd=WORK, env={**os.environ, **env}, stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                meta["exit_code"] = proc.wait(timeout=max(deadline - time.time(), 0))
                meta["timed_out"] = False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                meta["exit_code"], meta["timed_out"] = None, True
            tools = tool_calls(log, seg_off)
            meta["segments"].append(dict(start=round(seg_t - t, 1), seconds=round(time.time() - seg_t, 1),
                                         exit_code=meta["exit_code"], timed_out=meta["timed_out"], tools=tools))
            idle = idle + 1 if tools == 0 else 0  # 셀 수 없으면(None) 무작업으로 치지 않음 → CONTINUE_MAX 만
            left = deadline - time.time()
            if len(meta["segments"]) == 1:  # 공식 설정이었다면 여기서 끝났을 시각 (+cont 와 비교용)
                meta["first_exit_seconds"] = meta["segments"][0]["seconds"]
            if not cont:
                break
            reason = ("timed_out" if meta["timed_out"] else f"exit {meta['exit_code']}" if meta["exit_code"] != 0
                      else "min_left" if left < CONTINUE_MIN_LEFT else "idle" if idle >= CONTINUE_IDLE_MAX
                      else "max" if len(meta["segments"]) > CONTINUE_MAX else None)
            sid = sid or session_id(log)
            cmd = None if reason else continue_command(sid, left)
            if cmd is None:
                meta["continue_stop"] = reason or "no_session"
                break
            step(f"하네스가 스스로 끝남 (도구 {tools}개, 연속 무작업 {idle}) → 이어 가기 {len(meta['segments'])}회째 "
                 f"(남은 {left:.0f}s)")
    stop.set()
    meta["continues"] = len(meta["segments"]) - 1
    meta["harness_seconds"] = round(time.time() - t, 1)
    step(f"하네스 종료 (timed_out={meta['timed_out']}, exit={meta['exit_code']}, 이어 가기 {meta['continues']}회)")


def find_submission() -> Path | None:
    for p in [HOME / "submission" / "submission.csv", WORK / "workspaces" / "exp" / "best_submission" / "submission.csv",
              WORK / "submission" / "submission.csv"]:
        if p.exists():
            return p
    return None


def pack_workspace(dst: Path | None = None):
    """에이전트가 만든 코드·작은 파일 (20MB 넘는 파일과 데이터 링크는 뺌)."""
    def keep(t):
        if t.islnk() or t.issym() or "/.codex/" in f"/{t.name}/" or t.size > 20 * 2**20:
            return None
        return t

    with tarfile.open(dst or OUT / "workspace.tgz", "w:gz") as tar:
        tar.add(WORK, arcname="workspace", filter=keep)
        tar.add(HOME / "submission", arcname="home_submission", filter=keep)


def main():
    for d in (HOME / "data", HOME / "submission", WORK, OUT):
        d.mkdir(parents=True, exist_ok=True)
    meta = dict(CFG, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        try:  # 사양 측정이 실패해도 실행은 계속
            meta["node"] = node_spec()
        except Exception as e:  # noqa: BLE001
            meta["node"] = {"error": repr(e)[:500]}
        step(f"노드 사양: {json.dumps(meta['node'], ensure_ascii=False)}")
        run(meta)
    except BaseException as e:  # 중간에 실패해도 그때까지의 기록은 회수
        meta["runner_error"] = repr(e)[:2000]
        raise
    finally:
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if TRACER is not None:  # root span + 남은 span 을 회수 전에 보냄 (otel.jsonl 도 회수됨). root 가 실패해도 나머지는 보냄
            try:
                trace_root(meta)
            except Exception as e:  # noqa: BLE001 - trace 실패로 회수를 막지 않음
                meta["trace_error"] = repr(e)[:500]
            try:
                TRACER.close()
            except Exception as e:  # noqa: BLE001
                meta["trace_error"] = (meta.get("trace_error", "") + f" close: {e!r}")[:1000]
        (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        dst = Path(CFG["out"])
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUT, dst, dirs_exist_ok=True)
        step(f"회수 → {dst}")


def sync_progress(stop: threading.Event, meta: dict):
    """실행 중에도 2분마다 기록(프록시·형식 검사·하네스 출력)을 Volume 의 out/progress/ 에 복사 → fab mleb agent peek.
    제출 파일이 바뀌면 그 제출·작업 디렉터리 코드·그때까지의 meta 도 progress/ 에 둠: 컨테이너가 중간에 죽으면
    (AI Runtime 플랫폼 오류, 2026-11-13 leaf Claude Code +fm 3번) 마지막 회수가 안 돌아 제출이 사라짐 → 정리 작업이
    이 사본으로 채점 (시간 초과 실행이 그때까지 저장한 제출로 채점되는 것과 같게)."""
    dst = Path(CFG["out"]) / "progress"
    snap = None  # 마지막으로 복사한 제출 (경로, mtime, 크기)
    while not stop.wait(120):
        try:
            dst.mkdir(parents=True, exist_ok=True)
            with LOG_LOCK:  # 프록시·형식 검사가 줄을 쓰는 도중에 복사하면 마지막 줄이 잘림 → 메모리로 읽고 잠금 밖에서 씀
                logs = {f.name: f.read_bytes() for f in OUT.glob("*.jsonl")}
            for name, data in logs.items():
                (dst / name).write_bytes(data)
            for f in ("harness_stderr.txt",):
                if (OUT / f).exists():
                    (dst / f).write_bytes((OUT / f).read_bytes()[-200_000:])
            (dst / "elapsed.txt").write_text(f"{time.time() - T0:.0f}\n")
            if (sub := find_submission()) and (now := (str(sub), sub.stat().st_mtime, sub.stat().st_size)) != snap:
                data = sub.read_bytes()  # 에이전트가 쓰는 도중이면 다음 회차에 다시 (mtime·크기가 또 바뀜)
                part = dict(meta, submission_from=str(sub), submission_mtime=round(now[1], 2),
                            harness_seconds=round(time.time() - meta["harness_started_at"], 1),
                            snapshot_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                SNAP.mkdir(exist_ok=True)  # OUT 밖 (마지막 회수가 OUT 을 통째로 올림)
                (SNAP / "submission.csv").write_bytes(data)
                part["format_ok"], part["format_msg"] = check_submission(SNAP / "submission.csv")
                pack_workspace(SNAP / "workspace.tgz")
                shutil.copy(SNAP / "workspace.tgz", dst / "workspace.tgz")
                shutil.copy(SNAP / "submission.csv", dst / "submission.csv")
                (dst / "meta.json").write_text(json.dumps(part, indent=2, ensure_ascii=False, default=str))  # 제출 뒤에
                snap = now
        except Exception as e:  # noqa: BLE001 - 이 스레드가 죽으면 진행 기록이 멈춤 (meta 를 읽는 중 main 이 바꾸는 경우 등) → 다음 회차에
            print(f"[agent-runner] 진행 기록 복사 실패: {e!r}", flush=True)


def run(meta: dict):
    global TRACER
    with tarfile.open(CFG["data_tgz"]) as tar:
        tar.extractall(HOME / "data", filter="data")
    shutil.copy(SRC / "validate_submission.sh", HOME / "validate_submission.sh")
    (HOME / "validate_submission.sh").chmod(0o755)
    step("데이터 준비")

    pip(*ML_PACKAGES)
    install_harness()
    fm_text = prepare_fm(meta) if FM else ""  # pip_before 전: pip_added 는 에이전트가 직접 설치한 것만
    install_audit(meta)
    (OUT / "pip_before.txt").write_text(pip_list())
    step("패키지·하네스 설치")

    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    meta["identity"] = ws.current_user.me().user_name
    if TRACES_TABLE:  # 표에서 이 실행 찾기: trace_id (프록시 LLM span·root), session.id = 실행 키 (하네스 trace 포함)
        TRACER = Tracer(ws)
        meta["trace_id"] = TRACE_ID
    serve(make_proxy(ws), PROXY_PORT)
    serve(Validate, VALIDATE_PORT)
    prompt = HOME / "instructions.txt"
    prompt.write_text(instructions(fm_text))
    shutil.copy(prompt, OUT / "full_instructions.txt")
    cmd, env, log = harness_command(prompt)
    step(f"하네스 시작: {CFG['harness']} × {CFG['model']} ({TIME_LIMIT}s)")

    run_harness(cmd, env, log, meta)

    sub = find_submission()
    meta["submission_from"] = str(sub) if sub else None
    if sub:
        meta["submission_mtime"] = round(sub.stat().st_mtime, 2)  # 마지막으로 쓴 시각 (사후 검사의 tfm_at_submit)
        shutil.copy(sub, OUT / "submission.csv")
        meta["format_ok"], meta["format_msg"] = check_submission(sub)
    (OUT / "pip_after.txt").write_text(pip_list())
    pack_workspace()


if __name__ == "__main__":
    main()
