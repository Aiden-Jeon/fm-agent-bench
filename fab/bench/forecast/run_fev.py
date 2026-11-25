"""원격 (Databricks AI Runtime GPU, 내 신원) 에서 FM · 기존 방법으로 fev task 를 모든 window 에 돌림. fab 을 import 하지 않음.

    python run_fev.py                                     # 같은 폴더의 job.json 대로: 방법마다 venv → task 마다 worker
    python run_fev.py worker <방법> <task.json> <출력>     # (그 방법의 venv 안에서) (방법, task) 하나

fab fc fm (fab/bench/forecast/__init__.py) 이 job.json 과 함께 코드 묶음으로 올림. 방법마다 패키지 버전이 달라 venv 를 따로
(`uv venv` + `uv pip install fev==<FEV_VERSION> <패키지>`). docker/air 이미지는 안 씀 (실행마다 설치, docs/forecast.md 6).

방법 (docs/forecast.md 6 · 7)
  - fev 래퍼: autogluon/fev @ <fev_commit> 의 models/<래퍼>/model.py · requirements.txt 를 실행할 때 받음 (Apache-2.0)
  - 이 트랙의 FM: tabpfn_ts (fev 의 tabpfn-ts 래퍼 = TabPFN v2 회귀, 라이선스 수락 없음. v3.5 는 뺌, 2026-11-25 사용자),
    tabicl_ts (tabicl 의 TabICLForecaster, fev 래퍼가 없어 여기서)

출력 (job.json 의 out, Volume): results/<task>/<방법>/
  windows.json  window 마다 fev 의 compute_metrics (SQL · MASE · WQL · WAPE) + cutoff, task 평균 (fev evaluation_summary)
  last.csv      마지막 window 예측 (원래 id · timestamp · predictions · 분위수 열) → 정리 작업이 Agent 와 같은 채점기로 채점
  status.json   ok · 오류 · 시간 · 패키지 버전.  log.txt  worker 출력
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
T0 = time.time()

# 방법 키 → (fev 래퍼 폴더 또는 None, 생성자 인자, 더 깔 패키지). 래퍼 인자는 공개 결과의 model_kwargs 와 같게
METHODS = {
    "snaive": ("statsforecast", {"model": "seasonal_naive"}, []),
    "autoets": ("statsforecast", {"model": "auto_ets", "max_season_length": 200}, []),
    "chronos2": ("chronos-2", {"device": "cuda", "batch_size": 100}, []),
    "chronos_bolt": ("chronos", {"device": "cuda", "batch_size": 256}, []),
    "timesfm25": ("timesfm-2.5", {"device": "auto"}, []),
    "tabpfn_ts": ("tabpfn-ts", {}, []),  # 공개 결과 TabPFN-TS (v2) 와 같은 래퍼 · 인자 (tabpfn 7.1.1 · tabpfn-time-series 1.0.10)
    "tabicl_ts": (None, {}, ["tabicl[forecast]==2.2.0"]),
}
KEEP = {"fev", "statsforecast", "chronos-forecasting", "timesfm", "tabpfn", "tabpfn-time-series", "tabicl", "torch"}  # 고정을 안 푸는 패키지
RELAXED: dict[str, list[str]] = {}  # 방법 → 고정을 푼 보조 패키지 (make_venv)
VERSIONS = ["fev", "tabpfn", "tabpfn-time-series", "tabicl", "chronos-forecasting", "timesfm", "statsforecast", "torch",
            "datasets", "numpy", "pandas"]


def step(msg):
    print(f"[run_fev] {msg} ({time.time() - T0:.0f}s)", flush=True)


# ---------- 실행기 (바깥 파이썬) ----------

def find_uv() -> str:
    if (uv := shutil.which("uv")) or (uv := next((p for p in ("/usr/local/bin/uv", str(Path.home() / ".local/bin/uv"))
                                                  if os.path.isfile(p)), None)):
        return uv
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    return shutil.which("uv") or str(Path(sys.executable).parent / "uv")


def fetch(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        dst.write_bytes(r.read())


def make_venv(uv: str, key: str, job: dict) -> Path:
    wrapper, _, extra = METHODS[key]
    venv = Path(f"/tmp/fev_venv/{key}")
    reqs = [f"fev=={job['fev_version']}", *extra]
    if wrapper:
        base = f"https://raw.githubusercontent.com/autogluon/fev/{job['fev_commit']}/models/{wrapper}"
        fetch(f"{base}/model.py", HERE / "wrappers" / wrapper / "model.py")
        try:
            fetch(f"{base}/requirements.txt", HERE / "wrappers" / wrapper / "requirements.txt")
            reqs += ["-r", str(HERE / "wrappers" / wrapper / "requirements.txt")]
        except OSError:  # requirements 가 없는 래퍼
            pass
    shutil.rmtree(venv, ignore_errors=True)  # 같은 노드에서 다시 돌 때 (로컬 확인 등)
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    step(f"{key}: 설치 {' '.join(reqs)}")
    install = [uv, "pip", "install", "-q", "--python", str(venv / "bin/python")]
    if subprocess.run([*install, *reqs], timeout=3600).returncode:
        # 패키지 저장소에 고정 버전이 없을 때 (사내 프록시에 autogluon.timeseries==1.5.0 이 없었음, 2026-11-25 로컬):
        # 모델 패키지 (KEEP) 의 고정은 두고 나머지 (래퍼의 보조 패키지) 고정만 풀어 한 번 더. 푼 것은 install.json 에
        req_files = [reqs[i + 1] for i, r in enumerate(reqs) if r == "-r"]
        relaxed, out = [], [r for r in reqs if r != "-r" and r not in req_files]
        for f in req_files:
            for line in Path(f).read_text().splitlines():
                line = line.split("#")[0].strip()
                if not line:
                    continue
                name = line.split("==")[0].split("[")[0].strip().lower().replace("_", "-").replace(".", "-")
                if "==" in line and name not in KEEP:
                    relaxed.append(line)
                    line = line.split("==")[0]
                out.append(line)
        step(f"{key}: 고정 버전 설치 실패 → 보조 패키지 고정을 풀고 다시 {relaxed}")
        subprocess.run([*install, *out], check=True, timeout=3600)
        RELAXED[key] = relaxed
    return venv / "bin" / "python"


def run(job: dict):
    out = Path(job["out"])
    uv = find_uv()
    summary = {}
    for key in job["methods"]:
        try:
            py = make_venv(uv, key, job)
        except Exception as e:  # noqa: BLE001 - 설치 실패는 그 방법만 빼고 계속
            summary[key] = {t: dict(ok=False, error=f"설치 실패: {e!r}"[:2000]) for t in job["tasks"]}
            step(f"{key}: 설치 실패 {e!r}")
            continue
        for name, tdef in job["tasks"].items():
            work = Path("/tmp/fev_out") / slug(name) / key
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            (work / "task.json").write_text(json.dumps(tdef))
            t1 = time.time()
            env = dict(os.environ, TQDM_DISABLE="1")
            with open(work / "log.txt", "w") as log:
                try:
                    rc = subprocess.run([str(py), str(Path(__file__).resolve()), "worker", key, str(work / "task.json"),
                                         str(work)], stdout=log, stderr=subprocess.STDOUT, env=env,
                                        timeout=job.get("task_timeout", 4 * 3600)).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            st = json.loads((work / "status.json").read_text()) if (work / "status.json").exists() else \
                dict(ok=False, error=f"worker 종료 {rc} (log.txt)")
            st["wall_s"] = round(time.time() - t1, 1)
            st["relaxed_pins"] = RELAXED.get(key, [])
            (work / "status.json").write_text(json.dumps(st, indent=1))
            dst = out / "results" / slug(name) / key
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(work, dst)  # 끝날 때마다 Volume 으로 (작업이 중간에 죽어도 끝난 것은 남음)
            summary.setdefault(key, {})[name] = {k: st.get(k) for k in ("ok", "error", "wall_s")}
            step(f"{key} × {name}: {'ok' if st.get('ok') else 'FAIL'} {st['wall_s']}s")
    (out / "done.json").write_text(json.dumps(summary, indent=1))
    step("끝")


def slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------- worker (방법의 venv 안) ----------

def tabicl_ts(task):
    import datasets
    import fev
    from tabicl import TabICLForecaster

    fc = TabICLForecaster()
    levels = task.quantile_levels
    y = task.target_columns[0]
    out = []
    for w in task.iter_windows():
        past, future, _ = fev.convert_input_data(w, "pandas")  # 단변량 task 만 (driver.py 와 같음)
        known = task.known_dynamic_columns  # past dynamic 은 미래 값이 없어 뺌, static 은 안 씀 (TabPFN-TS 와 같음)
        ren = {task.id_column: "item_id", task.timestamp_column: "timestamp", y: "target"}
        ctx = past.rename(columns=ren)[["item_id", "timestamp", "target", *known]]
        fut = future.rename(columns=ren)[["item_id", "timestamp", *known]]
        pred = fc.predict_df(ctx, future_df=fut, quantiles=levels)
        rows = []
        for item in dict.fromkeys(ctx.item_id):
            g = pred.loc[item]
            rows.append({"predictions": g["target"].astype(float).tolist(),
                         **{str(q): g[q].astype(float).tolist() for q in levels}})
        out.append(datasets.DatasetDict({y: datasets.Dataset.from_list(rows)}))
    return out, {}


def fev_wrapper(key, task):
    import importlib.util

    import fev

    wrapper, kwargs, _ = METHODS[key]
    spec = importlib.util.spec_from_file_location(f"wrap_{key}", HERE / "wrappers" / wrapper / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    name = getattr(mod, next(n for n in dir(mod) if isinstance(getattr(mod, n), type)
                             and issubclass(getattr(mod, n), fev.ForecastingModel) and getattr(mod, n) is not fev.ForecastingModel))
    model = name(**kwargs)
    preds = model.fit_predict(task)
    return preds, dict(training_time_s=model.training_time, inference_time_s=model.inference_time,
                       trained_on=task.dataset_config in getattr(model, "trained_on_datasets", []))


def worker(key: str, task_json: str, out: str):
    import importlib.metadata as md

    import fev
    import pandas as pd
    from fev.metrics import get_metric

    out = Path(out)
    status = dict(ok=False, method=key, versions={})
    for p in VERSIONS:
        try:
            status["versions"][p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    try:
        task = fev.Task(**json.loads(Path(task_json).read_text()))
        t0 = time.time()
        preds, extra = tabicl_ts(task) if key == "tabicl_ts" else fev_wrapper(key, task)
        status["fit_predict_s"] = round(time.time() - t0, 1)
        status.update(extra)
        metrics = [get_metric(m) for m in [task.eval_metric, *task.extra_metrics]]
        rows = []
        for i, (p, w) in enumerate(zip(preds, task.iter_windows(), strict=True)):
            clean = task.clean_and_validate_predictions(p)
            rows.append(dict(window=i, cutoff=str(w.cutoff),
                             **w.compute_metrics(clean, metrics, task.seasonality, task.quantile_levels)))
            if i == task.num_windows - 1:  # 마지막 window 예측 → last.csv (원래 id, fev 의 시리즈 순서)
                past, future = w.get_input_data()
                col = clean[task.target_columns[0]]
                ids, ts = future[task.id_column], future[task.timestamp_column]
                frames = []
                for k in range(len(col)):
                    r = col[k]
                    frames.append(pd.DataFrame({"id": ids[k], "timestamp": ts[k], "predictions": r["predictions"],
                                                **{str(q): r[str(q)] for q in task.quantile_levels}}))
                pd.concat(frames, ignore_index=True).to_csv(out / "last.csv", index=False)
        summ = task.evaluation_summary(preds, model_name=key)
        status.update(ok=True, num_windows=task.num_windows,
                      task_mean={m: summ[m] for m in ("SQL", "MASE", "WQL", "WAPE") if m in summ})
        (out / "windows.json").write_text(json.dumps(rows, indent=1, default=str))
    except BaseException as e:  # noqa: BLE001 - 오류를 status 에 남기고 끝 (실행기가 다음 task 로)
        status["error"] = traceback.format_exc()[-4000:]
        print(status["error"], flush=True)
        if not isinstance(e, Exception):
            raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=1, default=str))


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(*sys.argv[2:5])
    else:
        run(json.loads((HERE / "job.json").read_text()))
