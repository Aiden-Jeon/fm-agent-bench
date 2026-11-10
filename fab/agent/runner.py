"""MLE-bench 에이전트 1회 실행 (Databricks 서버리스 GPU 작업 안에서, 전용 SP 신원으로 돎).

fab 을 import 하지 않음. 같은 디렉터리의 run.json 이 설정 (fab/agent/__init__.py 가 만듦):
  {key, cid, harness, model, alias, hours, continue, data_tgz, out, harness_versions, gateway, upstream_model,
   traces_table}

흐름 (MLE-bench agents/aide/start.sh 를 따름):
  1. public 데이터 풀기 → HOME/data, 지시문 = instructions.txt(경로만 바꿈) + 추가 안내 + 대회 description.md
  2. 일반 ML 패키지 설치 (TFM 없음), 하네스 설치
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

T0 = time.time()
SRC = Path(sys.argv[0]).resolve().parent
CFG = json.loads((SRC / "run.json").read_text())
HOME = Path("/tmp/mleb_home")  # MLE-bench 의 /home 자리 (data/, submission/, instructions.txt)
WORK = Path("/tmp/mleb_work")  # 하네스 작업 디렉터리 (에이전트 코드가 여기 남음)
OUT = Path("/tmp/mleb_out")
PROXY_PORT = 8765
VALIDATE_PORT = 5000
TIME_LIMIT = int(CFG["hours"] * 3600)

# 일반 ML 패키지 (MLE-bench environment/requirements.txt 중 표 형식에 쓰는 것). TFM(tabpfn·tabicl 등)은 넣지 않음
ML_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "hyperopt",
               "bayesian-optimization", "imbalanced-learn", "scikit-optimize", "matplotlib", "seaborn", "statsmodels",
               "pyarrow", "tqdm", "torch"]
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
#   - 실행 하나 = trace 하나: 끝날 때 root span (AGENT, 실행 설정·결과) 을 씀. LLM span 은 하네스와 상관없이 늘 그 아래
#     (meta.json 의 trace_id). 하네스가 traceparent 를 보냈으면 그 id 를 fab.harness_trace_id·fab.harness_parent_span_id 로 남김.
#     작업이 통째로 죽어 finally 가 안 돌면 root 가 없어 UI 에는 안 보이고 표에만 남음
#   - Claude Code 의 OTLP trace (interaction·llm_request·tool 실행 span) 도 프록시의 /otel/v1/traces 로 받아 같은 표로 넘김
#     (ug 의 otel-proxy 와 같은 방식: 하네스 설정에 고정 토큰을 넣으면 1시간 실행 중에 만료됨). 넘길 때 resource 에 fab.run
#     (실행 키) 을 붙임. Codex 는 끔 (2026-11-11 시험: 6분에 내부 Rust span 약 2.6만 개·작은 trace 수백 개, 도구·명령 내용은
#     없음 → 명령은 codex.jsonl, 요청·응답은 프록시 LLM span). AIDE 는 OTLP 를 안 내보냄
#   - 전송 기록은 otel.jsonl (proxy.jsonl 과 따로: 사후 검사가 proxy.jsonl 줄 수를 LLM 호출 수로 셈). 표에 쓰기 실패는 실행을 막지 않음
# gateway=true (선택): 같은 요청을 FM API 대신 Unity Gateway 로 보냄 → system.ai_gateway.usage 에 남음 (메타스토어에 Gateway
#   trace 표가 설정돼 있으면 거기에도). 모델은 model service 이름(system.ai.*, run.json 의 upstream_model). 요청마다
#   Databricks-Ai-Gateway-Request-Tags (실행 키·하네스·모델) 와 traceparent 를 붙임. x-databricks-use-coding-agent-mode 헤더는
#   일부러 안 넣음 (Smart Routing·코딩 에이전트 전용 처리가 붙을 수 있어 FM API 실행과 조건이 달라짐)
GATEWAY = bool(CFG.get("gateway"))
TRACES_TABLE = CFG.get("traces_table") or None
UPSTREAM_MODEL = CFG.get("upstream_model") or CFG["model"]  # 프록시가 고정하는 모델 (gateway 면 system.ai.*)
TRACE_ID, ROOT_SPAN_ID = os.urandom(16).hex(), os.urandom(8).hex()  # 이 실행의 trace·root span
REQUEST_TAGS = json.dumps({"fab_run": CFG["key"], "fab_harness": CFG["harness"], "fab_model": CFG["model"]})
OTEL_PATH, OTEL_TARGET = "/otel/v1/traces", "/api/2.0/otel/v1/traces"
TRACE_BODY_MAX = 1_000_000  # span 의 요청·응답 글자 수 상한
TRACE_FLUSH_SECS = 15
TRACE_BATCH_BYTES = 4_000_000

ROUTES = {  # 프록시 경로 → FM API 경로 (gateway 면 Unity Gateway 경로)
    "/anthropic/v1/messages": "/serving-endpoints/anthropic/v1/messages",
    "/openai/v1/chat/completions": "/serving-endpoints/chat/completions",
    "/openai/v1/responses": "/serving-endpoints/responses",
} if not GATEWAY else {
    "/anthropic/v1/messages": "/ai-gateway/anthropic/v1/messages",  # Anthropic 네이티브 (Claude 전용)
    "/openai/v1/chat/completions": "/ai-gateway/mlflow/v1/chat/completions",  # 통합 API (AIDE: Claude·GPT 둘 다)
    "/openai/v1/responses": "/ai-gateway/openai/v1/responses",  # OpenAI 네이티브 (GPT 전용)
}


def span_context(headers) -> tuple[str, str, str, dict]:
    """LLM 호출 span 의 (trace id, 부모 span id, 이 span id, 하네스 trace 속성). 늘 이 실행의 root 아래.
    하네스가 traceparent 를 보냈으면 (Claude Code) 그 trace·span id 를 속성으로 (하네스 span 과 맞춰 볼 때)."""
    tp = next((v for k, v in headers.items() if k.lower() == "traceparent"), "")
    m = re.fullmatch(r"[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}", tp.strip().lower())
    harness = {"harness_trace_id": m.group(1), "harness_parent_span_id": m.group(2)} if m else {}
    return TRACE_ID, ROOT_SPAN_ID, os.urandom(8).hex(), harness


def upstream_headers(headers, ctx) -> dict:
    """LLM 요청에 붙일 헤더 (gateway 일 때만): 요청 태그, (하네스가 안 보냈으면) 이 호출 span 의 traceparent."""
    if not GATEWAY:
        return {}
    extra = {"Databricks-Ai-Gateway-Request-Tags": REQUEST_TAGS}
    if not any(k.lower() == "traceparent" for k in headers):
        extra["traceparent"] = f"00-{ctx[0]}-{ctx[2]}-01"
    return extra


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


def tag_resource(body: bytes) -> bytes:
    """하네스 OTLP (ExportTraceServiceRequest) 의 ResourceSpans 마다 resource 에 fab.run 을 더함. protobuf 는 단일 메시지
    필드가 두 번 나오면 합치므로 (attributes 가 이어짐) ResourceSpans 끝에 resource(필드 1) 를 하나 더 붙이면 됨.
    최상위 필드만 훑음 (resource_spans 외 필드는 그대로)."""
    tag, out, i = _ld(1, _attrs(1, {"fab.run": CFG["key"]})), [], 0
    while i < len(body):
        start = i
        key, i = _read_varint(body, i)
        wire = key & 7
        if wire == 2:
            n, j = _read_varint(body, i)
            i = j + n
            if key >> 3 == 1:
                out.append(_ld(1, body[j:i] + tag))
                continue
        elif wire == 0:
            _, i = _read_varint(body, i)
        elif wire in (1, 5):
            i += 8 if wire == 1 else 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.append(body[start:i])
    return b"".join(out)


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


class Tracer:
    """프록시 LLM span 을 모아 TRACE_FLUSH_SECS 마다 traces_table 로 보냄. 실패해도 실행은 계속 (otel.jsonl 에 남김)."""

    def __init__(self, ws):
        self.ws, self.host = ws, urllib.parse.urlparse(ws.config.host).netloc
        self.spans, self.lock, self.stop = [], threading.Lock(), threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def add(self, span: bytes):
        with self.lock:
            self.spans.append(span)

    def _loop(self):
        while not self.stop.wait(TRACE_FLUSH_SECS):
            self.flush()

    def flush(self):
        with self.lock:
            spans, self.spans = self.spans, []
        batch, size = [], 0
        for sp in spans + [None]:
            if sp is None or (batch and size + len(sp) > TRACE_BATCH_BYTES):
                if batch:
                    self._post(batch)
                batch, size = [], 0
            if sp is not None:
                batch.append(sp)
                size += len(sp)

    def _post(self, batch: list[bytes]):
        body, t, status, err = otlp_request(batch), time.time(), None, None
        for attempt in range(3):
            conn = http.client.HTTPSConnection(self.host, timeout=120)
            try:
                conn.request("POST", OTEL_TARGET, body, {"Content-Type": "application/x-protobuf",
                                                         "X-Databricks-UC-Table-Name": TRACES_TABLE,
                                                         **self.ws.config.authenticate()})
                resp = conn.getresponse()
                status, data = resp.status, resp.read()
                err = data.decode("utf-8", "replace")[:300] if status >= 400 else None
            except Exception as e:  # noqa: BLE001 - trace 전송 실패는 실행을 막지 않음
                status, err = None, repr(e)[:300]
            finally:
                conn.close()
            if status is not None and status < 500 and status != 429:
                break
            time.sleep(2 * (attempt + 1))
        with LOG_LOCK, open(OUT / "otel.jsonl", "a") as f:
            f.write(json.dumps(dict(t=round(time.time() - T0, 1), source="fab", status=status, spans=len(batch),
                                    bytes=len(body), secs=round(time.time() - t, 1),
                                    **({"error": err} if err else {}))) + "\n")

    def close(self):
        self.stop.set()
        self.flush()


TRACER: Tracer | None = None  # run() 이 만듦 (traces_table 이 있을 때)


def trace_llm(ctx, path, t, req, text, status, asked, usage, extra):
    """LLM 호출 하나를 span 으로 (TRACER 가 있을 때). ctx = span_context()."""
    if TRACER is None:
        return
    fmt = "anthropic" if path.startswith("/anthropic") else "openai"
    tokens = {"input_tokens": sum(usage.get(k, 0) for k in ("input_tokens", "prompt_tokens", "cache_read_input_tokens",
                                                           "cache_creation_input_tokens")),
              "output_tokens": sum(usage.get(k, 0) for k in ("output_tokens", "completion_tokens"))}
    tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
    attrs = {**mlflow_attrs("LLM", ctx[0], body_value(json.dumps(req, ensure_ascii=False)), body_value(text)),
             "mlflow.llm.model": UPSTREAM_MODEL, "mlflow.message.format": fmt, "mlflow.chat.tokenUsage": tokens,
             "session.id": CFG["key"], "gen_ai.request.model": UPSTREAM_MODEL,
             "gen_ai.usage.input_tokens": tokens["input_tokens"], "gen_ai.usage.output_tokens": tokens["output_tokens"],
             "http.response.status_code": status, "fab.path": path, "fab.asked_model": asked,
             "fab.route": "gateway" if GATEWAY else "fm_api", **{f"fab.{k}": v for k, v in {**ctx[3], **extra}.items()}}
    error = f"HTTP {status}" if not status or status >= 400 else None
    TRACER.add(otlp_span(ctx[0], ctx[1], ctx[2], f"llm {path}", t, time.time(), attrs, error, kind=3))


def trace_root(meta: dict):
    """실행 전체 span (끝날 때 한 번). LLM span 의 부모."""
    if TRACER is None:
        return
    keys = ("cid", "harness", "model", "alias", "hours", "continue", "gateway", "upstream_model")
    outs = ("timed_out", "harness_seconds", "format_ok", "format_msg", "submission_from", "continues", "runner_error")
    attrs = {**mlflow_attrs("AGENT", TRACE_ID, {k: CFG.get(k) for k in keys}, {k: meta.get(k) for k in outs}),
             "session.id": CFG["key"],
             "fab.run": CFG["key"], **{f"fab.{k}": CFG.get(k) for k in keys if not isinstance(CFG.get(k), (dict, list))}}
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


def message_text(content) -> str:
    """Chat Completions 의 content: 문자열, 또는 파트 목록 (긴 프롬프트에서 Claude 가 [{type: text, text}, ...] 로 답함)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return content or ""


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
                    body = json.dumps({"error": {"message": f"fab proxy error: {e!r}"[:500]}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
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
            if path == "/openai/v1/chat/completions" and (conv := tools_as_json_prompt(req)):
                return self.tool_call(target, *conv, asked, dropped, t, ctx)
            # accept-encoding 도 뺌: 넘기면 FM API 가 gzip 으로 답하는데 프록시는 content-encoding 헤더를 지우고
            # 본문만 넘겨서 하네스가 못 읽음 (에러 응답이 깨지고, 사용량·에러 기록도 gzip 바이트가 됨).
            # content-encoding 도 뺌: 본문은 위에서 풀어 다시 쓴 JSON
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "authorization", "x-api-key", "content-length", "connection",
                                            "accept-encoding", "content-encoding")}
            headers |= {"Content-Length": str(len(body)), **upstream_headers(headers, ctx)}
            tail, status, waited, retries, conn = b"", None, 0, 0, None
            try:
                conn, resp, retries, waited = self.send(target, body, headers)
                status = resp.status
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding"):
                        self.send_header(k, v)
                self.end_headers()
                while chunk := resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    tail = (tail + chunk)[-200_000:]
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if conn:
                    conn.close()
            text = tail.decode("utf-8", "replace")
            usage = {k: max(map(int, v)) for k in USAGE_KEYS if (v := re.findall(rf'"{k}"\s*:\s*(\d+)', text))}
            err = text[:500] if status and status >= 400 else None
            self.record(dict(path=path, asked=asked, status=status, secs=round(time.time() - t, 1), **usage, **sent,
                             **({"error": err} if err else {}), **({"dropped": dropped} if dropped else {}),
                             **({"retries_429": retries, "waited": round(waited, 1)} if retries else {})))
            trace_llm(ctx, path, t, req, text, status, asked, usage,
                      dict(sent, retries_429=retries, waited_429=round(waited, 1), dropped=",".join(dropped) or None))

        def reject(self, status: int, path: str, message: str):
            body = json.dumps({"error": {"message": message[:500]}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
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
            conn, resp, retries, waited = self.send(target, body, headers | upstream_headers(headers, ctx))
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
            self.send_response(status or 502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
            self.record(dict(path=self.path, asked=asked, status=status, secs=round(time.time() - t, 1),
                             tool_json=fn["name"], tool_tries=tries, tool_ok=obj is not None,
                             **{k: v for k, v in usage.items() if k in USAGE_KEYS},
                             **({"error": err} if err else {}), **({"dropped": dropped} if dropped else {}),
                             **({"retries_429": retries, "waited": round(waited, 1)} if retries else {})))
            trace_llm(ctx, self.path, t, req, data.decode("utf-8", "replace"), status, asked, usage,
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
                body = tag_resource(body)
            except (ValueError, IndexError):  # protobuf 가 아니거나 깨짐 → 그대로 넘김 (서버가 거부하면 otel.jsonl 에 남음)
                pass
            headers = {"Content-Type": self.headers.get("Content-Type", "application/x-protobuf"),
                       "Content-Length": str(len(body)), "X-Databricks-UC-Table-Name": TRACES_TABLE}
            conn, resp, retries, waited = self.send(OTEL_TARGET, body, headers)
            try:
                data = resp.read()
            finally:
                conn.close()
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type") or "application/x-protobuf")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            # 200 이어도 일부 span 이 거부될 수 있음 (partial_success, protobuf) → 응답 크기만 남기고 내용은 안 풂
            self.record(dict(source="harness", status=resp.status, bytes=len(body), resp_bytes=len(data),
                             secs=round(time.time() - t, 1),
                             **({"error": data.decode("utf-8", "replace")[:300]} if resp.status >= 400 else {}),
                             **({"retries_429": retries, "waited": round(waited, 1)} if retries else {})), "otel.jsonl")

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

def instructions() -> str:
    # GPU 이름만 (P0 의 --query-gpu=name 출력과 같은 글: 지시문이 실행마다 같도록)
    hardware = "\n".join(line.split(",")[0].strip() for line in gpu_info() or ())
    h, m = divmod(TIME_LIMIT // 60, 60)
    # AIDE 는 자기 작업 공간에서 돌아 MLE-bench start.sh 처럼 /home/ 을 지워 상대 경로로 (submission/submission.csv)
    text = (SRC / "instructions.txt").read_text().replace("/home/", "" if CFG["harness"] == "aide" else f"{HOME}/")
    notes = (SRC / "additional_notes.txt").read_text().format(
        hardware=f"1 {hardware}" if hardware else "a CPU", time_limit=f"{h}hrs {m}mins 0secs")
    desc = (HOME / "data" / "description.md").read_text()
    return f"{text}\n\n{notes}\n\nCOMPETITION INSTRUCTIONS\n------\n\n{desc}"


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
    env = {"PATH": f"{Path.home() / '.local' / 'bin'}:{os.environ['PATH']}"}
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
            # 토큰은 프록시가 붙임 → otelHeadersHelper 불필요. PROPAGATE_TRACEPARENT: LLM 요청에 traceparent 를 붙여
            # (trace linking 이 켜져 있으면) Gateway span 이 Claude Code 의 trace 에 묶임
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
# 모델이 매번 바로 끝내도 남은 시간 내내 호출하지 않도록 CONTINUE_MAX 번까지만.
CONTINUE_MIN_LEFT = 120  # 초
CONTINUE_MAX = 20
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
    이어 가기를 멈춘 까닭은 meta["continue_stop"] 에 남김."""
    t = time.time()
    deadline = t + TIME_LIMIT
    stop = threading.Event()
    threading.Thread(target=sync_progress, args=(stop,), daemon=True).start()
    cont, sid, meta["segments"] = bool(CFG.get("continue")), None, []
    # 이어 가기 구간도 같은 출력 파일에 덧붙임 → 사후 검사·토큰 집계가 전체를 봄
    with open(log, "w") as out, open(OUT / "harness_stderr.txt", "w") as err:
        while True:
            seg_t = time.time()
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
            meta["segments"].append(dict(start=round(seg_t - t, 1), seconds=round(time.time() - seg_t, 1),
                                         exit_code=meta["exit_code"], timed_out=meta["timed_out"]))
            left = deadline - time.time()
            if len(meta["segments"]) == 1:  # 공식 설정이었다면 여기서 끝났을 시각 (+cont 와 비교용)
                meta["first_exit_seconds"] = meta["segments"][0]["seconds"]
            if not cont:
                break
            reason = ("timed_out" if meta["timed_out"] else f"exit {meta['exit_code']}" if meta["exit_code"] != 0
                      else "min_left" if left < CONTINUE_MIN_LEFT else "max" if len(meta["segments"]) > CONTINUE_MAX
                      else None)
            sid = sid or session_id(log)
            cmd = None if reason else continue_command(sid, left)
            if cmd is None:
                meta["continue_stop"] = reason or "no_session"
                break
            step(f"하네스가 스스로 끝남 → 이어 가기 {len(meta['segments'])}회째 (남은 {left:.0f}s)")
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


def pack_workspace():
    """에이전트가 만든 코드·작은 파일 (20MB 넘는 파일과 데이터 링크는 뺌)."""
    def keep(t):
        if t.islnk() or t.issym() or "/.codex/" in f"/{t.name}/" or t.size > 20 * 2**20:
            return None
        return t

    with tarfile.open(OUT / "workspace.tgz", "w:gz") as tar:
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
        if TRACER is not None:  # root span + 남은 LLM span 을 회수 전에 보냄 (otel.jsonl 도 회수됨)
            try:
                trace_root(meta)
                TRACER.close()
            except Exception as e:  # noqa: BLE001 - trace 실패로 회수를 막지 않음
                meta["trace_error"] = repr(e)[:500]
        (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        dst = Path(CFG["out"])
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUT, dst, dirs_exist_ok=True)
        step(f"회수 → {dst}")


def sync_progress(stop: threading.Event):
    """실행 중에도 2분마다 기록(프록시·형식 검사·하네스 출력)을 Volume 의 out/progress/ 에 복사 → fab mleb agent peek."""
    dst = Path(CFG["out"]) / "progress"
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
        except OSError as e:
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
    prompt.write_text(instructions())
    shutil.copy(prompt, OUT / "full_instructions.txt")
    cmd, env, log = harness_command(prompt)
    step(f"하네스 시작: {CFG['harness']} × {CFG['model']} ({TIME_LIMIT}s)")

    run_harness(cmd, env, log, meta)

    sub = find_submission()
    meta["submission_from"] = str(sub) if sub else None
    if sub:
        shutil.copy(sub, OUT / "submission.csv")
        meta["format_ok"], meta["format_msg"] = check_submission(sub)
    (OUT / "pip_after.txt").write_text(pip_list())
    pack_workspace()


if __name__ == "__main__":
    main()
