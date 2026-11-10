"""원격 상태와 정리 작업 (로컬 쪽). 학습·에이전트는 제출한 뒤 맥북을 꺼도 원격에서 계속 돌고, 결과 합치기·채점은
원할 때 정리 작업을 한 번 돌려 처리 (스케줄 없음, 2026-11-11 결정).

    uv run --frozen fab remote deploy                # 정리 작업(serverless CPU job) 만들기·갱신, 결과표 초기값 올리기
    uv run --frozen fab remote status                # 처리 대기 항목, 최근 정리 작업 실행
    uv run --frozen fab remote kick                  # 정리 작업 한 번 실행 (끝난 실행만 처리, 아직 도는 것은 다음에)
    uv run --frozen fab sync                         # 원격 결과표 → results/*.csv (fab/remote/sync.py)

흐름: 로컬 명령(fab databricks, fab mleb grade, fab mleb agent run ...)은 제출만 하고 Volume 의 state/<종류>/<key>.json 에
제출 기록을 씀 → `fab remote kick` 으로 정리 작업(fab/remote/reconcile.py, 내 신원)을 돌리면 끝난 실행을 처리하고
state/<종류>/<key>.done.json 을 남김. 아직 끝나지 않은 실행·그 모델의 채점 요청은 그대로 남아 다음 kick 때 처리.

Volume (databricks.local.yaml 의 volume) 구조:
    state/{fm,agent,grade,exclude,recheck,batch}/   제출 기록과 처리 표시
    fab/code.tgz                                   정리 작업이 쓰는 fab 코드 (로컬 명령마다 지금 코드로 다시 올림)
    mlebench/code.tgz, mlebench/data/<cid>/         채점 코드(mlebench 패키지·리더보드)와 정답 (fab/bench/mlebench/remote.py)
    outputs/                                       예측·OOF (로컬 results/<대회>/<모델>/ 자리)
    tables/{results,mlebench,mlebench_runs}.csv    결과표 원본 (fab sync 가 results/ 로 받음)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from fab.core.competition import ROOT

LOCAL_CONFIG = ROOT / "databricks.local.yaml"
JOB_NAME = "fab-reconcile"
TABLES = ["results.csv", "mlebench.csv", "mlebench_runs.csv"]  # 원격이 원본인 표 (mlebench_agents.csv 는 로컬 fab mleb agents)
KINDS = ["fm", "agent", "grade", "exclude", "recheck"]
HERE = Path(__file__).parent


def settings() -> dict:
    cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
    if not (cfg.get("profile") and cfg.get("volume")):
        raise SystemExit(f"{LOCAL_CONFIG.name} 에 profile 과 volume 필요 (README 의 준비 참고)")
    return dict(profile=cfg["profile"], volume=cfg["volume"].rstrip("/"))


def dbx(cfg, *args, check=True, timeout=600):
    from fab.runner.databricks import databricks

    return databricks(cfg["profile"], *args, capture=True, check=check, timeout=timeout)


def now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


# --- Volume 의 제출 기록 ---------------------------------------------------------------------------------------


def put_json(cfg, rel: str, obj: dict):
    remote = f"{cfg['volume']}/{rel}"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / Path(rel).name
        f.write_text(json.dumps(obj, indent=2, default=str))
        dbx(cfg, "fs", "mkdir", f"dbfs:{remote.rsplit('/', 1)[0]}")  # 파일 cp 는 상위 폴더를 만들지 않음
        dbx(cfg, "fs", "cp", "--overwrite", str(f), f"dbfs:{remote}")


def read_json(cfg, rel: str) -> dict | None:
    res = dbx(cfg, "fs", "cat", f"dbfs:{cfg['volume']}/{rel}", check=False)
    return json.loads(res.stdout) if res.returncode == 0 else None


def list_names(cfg, rel: str) -> list[str]:
    res = dbx(cfg, "fs", "ls", f"dbfs:{cfg['volume']}/{rel}", check=False)
    return res.stdout.split() if res.returncode == 0 else []


def items(cfg, kind: str) -> dict[str, bool]:
    """{key: 처리됨 여부} (state/<kind>/<key>.json 과 <key>.done.json)."""
    names = set(list_names(cfg, f"state/{kind}"))
    keys = sorted(n.removesuffix(".json") for n in names if n.endswith(".json") and not n.endswith((".done.json",
                                                                                                    ".attempts.json")))
    return {k: f"{k}.done.json" in names for k in keys}


def submit_item(cfg, kind: str, key: str, manifest: dict):
    """제출 기록을 Volume 에 씀. 처리는 `fab remote kick` 으로 정리 작업을 돌릴 때."""
    put_json(cfg, f"state/{kind}/{key}.json", dict(manifest, kind=kind, key=key, submitted=now()))


# --- 코드 묶음 ---------------------------------------------------------------------------------------------------


def upload_code(cfg):
    """fab/, competitions/ (data·folds 캐시 제외는 원격 실행 스냅샷과 같은 규칙) → Volume 의 fab/code.tgz."""
    from fab.runner.common import CODE_PATHS

    with tempfile.TemporaryDirectory() as d:
        tgz = Path(d) / "code.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            for p in CODE_PATHS:
                tar.add(ROOT / p, arcname=p,
                        filter=lambda t: None if {"__pycache__", "data"} & set(Path(t.name).parts) else t)
        dbx(cfg, "fs", "mkdir", f"dbfs:{cfg['volume']}/fab")
        dbx(cfg, "fs", "cp", "--overwrite", str(tgz), f"dbfs:{cfg['volume']}/fab/code.tgz")


# --- 정리 작업 (Jobs) ---------------------------------------------------------------------------------------------


def job_id(cfg) -> int | None:
    res = dbx(cfg, "jobs", "list", "--name", JOB_NAME, "-o", "json")
    jobs = json.loads(res.stdout or "[]")
    jobs = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    return jobs[0]["job_id"] if jobs else None


def job_settings(cfg, me: str) -> dict:
    """스케줄 없음: `fab remote kick` 으로만 돎. 동시에 하나만 (같은 항목을 두 번 처리하지 않게)."""
    from fab.runner.databricks import CPU_ENV, requirement_lines

    return {
        "name": JOB_NAME, "max_concurrent_runs": 1, "timeout_seconds": 3 * 3600,
        "tasks": [{"task_key": "reconcile", "environment_key": "fab", "max_retries": 0,
                   "spark_python_task": {"python_file": f"/Workspace/Users/{me}/.fab/{JOB_NAME}/entry.py",
                                         "parameters": [cfg["volume"]]}}],
        # 패키지는 CPU 학습 작업과 같은 버전 (docker/air/requirements.txt) + Jobs 조회용 SDK
        "environments": [{"environment_key": "fab", "spec": {
            "environment_version": CPU_ENV,
            "dependencies": requirement_lines({"numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "databricks-sdk"})}}],
    }


def deploy(cfg):
    upload_code(cfg)
    seed_tables(cfg)
    me = json.loads(dbx(cfg, "current-user", "me", "-o", "json").stdout)["userName"]
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(HERE / "entry.py", d)
        dbx(cfg, "workspace", "import-dir", "--overwrite", d, f"/Workspace/Users/{me}/.fab/{JOB_NAME}")
    new = job_settings(cfg, me)
    if (jid := job_id(cfg)) is None:
        jid = json.loads(dbx(cfg, "jobs", "create", "--json", json.dumps(new), "-o", "json").stdout)["job_id"]
        print(f"정리 작업 생성: {JOB_NAME} (job_id {jid}, 스케줄 없음 → fab remote kick)")
    else:
        dbx(cfg, "jobs", "reset", "--json", json.dumps({"job_id": jid, "new_settings": new}))
        print(f"정리 작업 갱신: {JOB_NAME} (job_id {jid}, 스케줄 없음 → fab remote kick)")


def seed_tables(cfg):
    """원격 결과표가 없으면 로컬 표를 초기값으로 올림 (이미 있으면 그대로: 원격이 원본)."""
    from fab.core.results import RESULTS

    have = set(list_names(cfg, "tables"))
    for name in TABLES:
        if name in have or not (RESULTS / name).exists():
            continue
        dbx(cfg, "fs", "mkdir", f"dbfs:{cfg['volume']}/tables")
        dbx(cfg, "fs", "cp", str(RESULTS / name), f"dbfs:{cfg['volume']}/tables/{name}")
        print(f"결과표 초기값 업로드: {RESULTS / name} → {cfg['volume']}/tables/{name}")


def status(cfg):
    jid = job_id(cfg)
    if jid is None:
        print(f"정리 작업({JOB_NAME}) 없음 → fab remote deploy")
    else:
        print(f"{JOB_NAME} (job_id {jid}) 최근 실행:")
        runs = json.loads(dbx(cfg, "jobs", "list-runs", "--job-id", str(jid), "--limit", "3", "-o", "json").stdout
                          or "[]")
        for r in runs if isinstance(runs, list) else runs.get("runs", []):
            st = r.get("state", {})
            start = datetime.fromtimestamp(r.get("start_time", 0) / 1000).strftime("%m-%d %H:%M")
            print(f"  {start} {st.get('life_cycle_state')} {st.get('result_state') or ''}  {r.get('run_page_url')}")
    for kind in KINDS:
        its = items(cfg, kind)
        pending = [k for k, done in its.items() if not done]
        print(f"{kind}: 전체 {len(its)}, 대기 {len(pending)}" + "".join(f"\n  {k}" for k in pending))


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="fab remote")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy", help="정리 작업 만들기·갱신 (코드·결과표 초기값도 올림)")
    sub.add_parser("status", help="처리 대기 항목·최근 정리 작업 실행")
    sub.add_parser("kick", help="정리 작업을 한 번 실행 (끝난 실행만 처리)")
    args = ap.parse_args(argv)
    cfg = settings()
    if args.cmd == "deploy":
        deploy(cfg)
    elif args.cmd == "status":
        status(cfg)
    else:
        if (jid := job_id(cfg)) is None:
            raise SystemExit(f"정리 작업({JOB_NAME}) 없음 → fab remote deploy")
        upload_code(cfg)  # 정리 작업은 지금 로컬 코드로 돎
        res = json.loads(dbx(cfg, "jobs", "run-now", str(jid), "--no-wait", "-o", "json").stdout)
        print(f"실행: run_id {res.get('run_id')}")
