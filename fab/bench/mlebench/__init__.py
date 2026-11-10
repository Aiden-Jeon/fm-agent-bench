"""MLE-bench 연동: 대회 준비(MLE-bench 의 train/test 분할, 로컬), 채점(원격 정리 작업, remote.py·fab/remote).
계획은 docs/plans/mle-bench.md.

MLE-bench 는 원래 train 을 다시 나눠 자체 test(정답 비공개)를 만들고 그 test 로 채점함 → 공개 리더보드 에이전트와
비교하려면 TFM 도 같은 분할로 학습·채점해야 함 (Kaggle LB 가 아님).

mlebench 는 kaggle<1.7·tensorflow 를 요구해 이 프로젝트와 같이 못 깖 → mle-bench 저장소를 옆에 두고 자체 가상환경에서 돌림:

    git clone https://github.com/openai/mle-bench ../mle-bench      # LFS 없이 받아도 됨 (필요한 csv 는 fab 이 받음), LFS 로 받아도 됨
    uv venv --python 3.11 ../mle-bench/.venv
    uv pip install --python ../mle-bench/.venv/bin/python --no-deps -e ../mle-bench
    uv pip install --python ../mle-bench/.venv/bin/python appdirs "pandas>=2.2,<2.3" "numpy<2.1" pyyaml py7zr \
        "scikit-learn>=1.5,<1.6" tqdm diskcache tenacity

위치는 FAB_MLEBENCH 로 바꿈. 로컬 mle-bench 는 분할 준비(prepare)·데이터 위치(paths)에만 씀.
대회 data/ 에는 public 만 복사 (data/ 는 원격 실행·에이전트에 올라가는 곳). 정답(private)은 mlebench 캐시와
채점용으로 올린 내 Volume 의 mlebench/data/ 에만 있음 (에이전트 SP 의 scratch Volume 에는 없음, remote.py).
"""

from __future__ import annotations

import argparse
import json
import io
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from fab.core.competition import COMP_DIR, ROOT, Competition, load_competition
from fab.core.results import OUTPUTS, RESULTS, append_rows, test_info

MLEBENCH_DIR = Path(os.environ.get("FAB_MLEBENCH", ROOT.parent / "mle-bench")).expanduser().resolve()
DRIVER = Path(__file__).with_name("driver.py")
LFS_URL = "https://media.githubusercontent.com/media/openai/mle-bench/main/"
GRADES = RESULTS / "mlebench.csv"  # fab mleb grade 결과 (git 추적)
AGENTS = RESULTS / "mlebench_agents.csv"  # fab mleb agents: 공개 에이전트 실행별 점수 (TABULAR 대회만, git 추적)
LFS_BATCH = "https://github.com/openai/mle-bench.git/info/lfs/objects/batch"
# 비교 대상 표 형식 대회 (docs/plans/mle-bench.md). dev 세트(spaceship-titanic, playground-series-s3e18)는 공개 결과에 없음
TABULAR = ["spaceship-titanic", "playground-series-s3e18", "nomad2018-predict-transparent-conductors",
           "tabular-playground-series-dec-2021", "tabular-playground-series-may-2022",
           "new-york-city-taxi-fare-prediction", "leaf-classification"]


def _python() -> Path:
    py = MLEBENCH_DIR / ".venv" / "bin" / "python"
    if not py.exists():
        sys.exit(f"mle-bench 가상환경 없음: {py}\n설치 방법은 fab/bench/mlebench/__init__.py 위쪽 설명 참고 (FAB_MLEBENCH 로 위치 지정)")
    return py


def _driver(*args: str) -> dict:
    # cwd: mlebench 가 체크섬 캐시(cache/)를 cwd 에 만듦 → 이 저장소가 아니라 mle-bench 쪽에
    out = subprocess.run([str(_python()), "-I", str(DRIVER), *args], stdout=subprocess.PIPE, text=True, check=True,
                         cwd=MLEBENCH_DIR)
    return json.loads(out.stdout.strip().splitlines()[-1])


def fetch_lfs(path: Path):
    """LFS 포인터 파일이면 실제 파일로 바꿈 (mle-bench 를 LFS 없이 받았을 때)."""
    if path.stat().st_size > 1024 or not path.read_bytes().startswith(b"version https://git-lfs"):
        return
    url = LFS_URL + path.relative_to(MLEBENCH_DIR).as_posix()
    print(f"LFS 받기: {url}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def lfs_objects(pointer: str) -> list[dict]:
    """LFS 포인터 안의 객체들. 병합 충돌이 난 포인터(runs/run_group_experiments.csv)는 객체가 여럿."""
    oids = re.findall(r"oid sha256:([0-9a-f]{64})", pointer)
    sizes = re.findall(r"^size (\d+)", pointer, re.M)
    return [dict(oid=o, size=int(n)) for o, n in zip(oids, sizes)]


def fetch_lfs_objects(objects: list[dict]) -> dict[str, bytes]:
    """LFS batch API 로 객체를 받아 {oid: 내용}. media URL 은 충돌 난 포인터를 못 풀어서 이쪽을 씀."""
    out = {}
    for i in range(0, len(objects), 100):
        req = urllib.request.Request(LFS_BATCH, method="POST", headers={
            "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"},
            data=json.dumps(dict(operation="download", transfers=["basic"], objects=objects[i:i + 100])).encode())
        with urllib.request.urlopen(req, timeout=60) as r:
            for o in json.load(r)["objects"]:
                with urllib.request.urlopen(o["actions"]["download"]["href"], timeout=120) as f:
                    out[o["oid"]] = f.read()
    return out


def resolve_lfs(pointers: dict) -> dict[object, list[bytes]]:
    """{키: LFS 포인터 문자열} → {키: [실제 내용, ...]}. 받은 객체가 또 포인터면 (충돌 난 포인터를 그대로 LFS 에
    올린 경우, runs/run_group_experiments.csv) 끝까지 풂. 포인터가 아니면 (LFS 로 받은 저장소) 그 내용 그대로."""
    plain = {k: [v.encode()] for k, v in pointers.items() if not v.startswith("version https://git-lfs")}
    objs = {k: lfs_objects(v) for k, v in pointers.items() if k not in plain}
    blobs = fetch_lfs_objects(list({o["oid"]: o for os_ in objs.values() for o in os_}.values())) if objs else {}
    out = dict(plain)
    for k, os_ in objs.items():
        out[k] = []
        for o in os_:
            b = blobs[o["oid"]]
            out[k] += resolve_lfs({0: b.decode()})[0] if b.startswith(b"version https://git-lfs") else [b]
    return out


def ensure_lfs(cid: str):
    d = MLEBENCH_DIR / "mlebench" / "competitions" / cid
    if not d.is_dir():
        sys.exit(f"MLE-bench 에 없는 대회: {cid} ({d})")
    for p in d.glob("*.csv"):  # leaderboard.csv (메달 기준선·순위)
        fetch_lfs(p)


def prepare(comp: Competition):
    """fab download 가 부름: Kaggle zip 받기 → MLE-bench 분할 (체크섬 확인) → public 을 comp.data_dir 에 복사."""
    cid = comp.mlebench
    ensure_lfs(cid)
    paths = _driver("paths", cid)
    zip_path = Path(paths["dir"]) / f"{cid}.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if subprocess.run(["kaggle", "competitions", "download", "-c", cid, "-p", str(zip_path.parent)]).returncode:
            sys.exit(f"{cid} 다운로드 실패. 403 이면 Rules 동의 필요: https://www.kaggle.com/competitions/{cid}/rules")
    paths = _driver("prepare", cid, str(zip_path))
    comp.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(paths["public"], comp.data_dir, dirs_exist_ok=True)
    print(f"MLE-bench {cid}: public → {comp.data_dir} ({sorted(p.name for p in comp.data_dir.iterdir())})")


def bench_comps(cid: str) -> list[Competition]:
    """MLE-bench 대회 하나를 맡는 fab 대회들. 타깃이 여럿이면 (nomad2018) 타깃마다 fab 대회 하나 → 제출 때 열을 합침."""
    comps = [load_competition(p.parent.name) for p in sorted(COMP_DIR.glob("*/config.yaml"))]
    return [c for c in comps if c.mlebench == cid]


def make_submission(cid: str, models: list[str]) -> Path:
    from fab.submission import make_submission as comp_submission

    comps = bench_comps(cid)
    if not comps:
        sys.exit(f"config.yaml 에 mlebench: {cid} 인 대회가 없음")
    subs = [pd.read_csv(comp_submission(c.name, models)) for c in comps]
    sub = subs[0]
    for c, s in zip(comps[1:], subs[1:]):  # 같은 test (같은 id 순서) → 타깃 열만 붙임
        id_col = s.columns[0]
        if not s[id_col].equals(sub[id_col]):
            sys.exit(f"{c.name} 의 제출 id 가 {comps[0].name} 과 다름")
        sub[s.columns[1]] = s[s.columns[1]]
    for c, col in zip(comps, sub.columns[1:]):
        # log1p 타깃 = RMSLE 채점. log 공간 예측이 0 살짝 아래면 expm1 < 0 → mlebench 채점기(sklearn 1.5
        # root_mean_squared_log_error)가 ValueError → score=None 이 됨 (nomad2018 formation 은 타깃이 0 근처)
        if c.target_transform == "log1p":
            sub[col] = sub[col].clip(lower=0)
    path = OUTPUTS / "_mlebench" / cid / f"{'+'.join(models)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def grade(cid: str, groups: list[list[str]], note: str, grade_fn) -> list[dict]:
    """모델 묶음마다 제출 파일을 만들어 (예측 평균) grade_fn(cid, csv) 로 채점 → results/mlebench.csv.
    원격 정리 작업이 부름 (grade_fn = remote.Grader.grade). 예측이 없거나 채점에 실패한 묶음은 표에 쓰지 않고 오류만 출력."""
    comps = bench_comps(cid)
    rows = []
    for models in groups:
        if missing := [f"{c.name}/{m}" for c in comps for m in models if test_info(c.name, m) is None]:
            print(f"{cid} {'+'.join(models)}: test 예측 없음 ({', '.join(missing)}) → 채점 안 함")
            continue
        report = grade_fn(cid, make_submission(cid, models))
        if report.get("error"):
            print(f"{cid} {'+'.join(models)}: 채점 실패\n{report['error']}")
            continue
        infos = [test_info(c.name, m) or {} for c in comps for m in models]
        rows.append(dict(timestamp=pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds"),
                         mlebench=cid, model="+".join(models), comps=" ".join(c.name for c in comps),
                         runs=" ".join(str(i.get("run")) for i in infos), note=note,
                         **{k: report.get(k) for k in ["score", "valid_submission", "any_medal", "gold_medal",
                                                       "silver_medal", "bronze_medal", "above_median",
                                                       "gold_threshold", "silver_threshold", "bronze_threshold",
                                                       "median_threshold", "is_lower_better", "lb_rank", "lb_teams"]}))
        r = rows[-1]
        rank = f"  LB {r['lb_rank']}/{r['lb_teams']}" if r.get("lb_rank") else ""
        print(f"{cid} {r['model']}: score={r['score']}  {medal(r)}{rank}  "
              f"(금 {r['gold_threshold']}, 중앙값 {r['median_threshold']})")
    if rows:
        append_rows(pd.DataFrame(rows), GRADES)
    return rows


def request_grade(cid: str, groups: list[list[str]], note: str = ""):
    """채점 요청을 Volume 에 남김 → 원격 정리 작업이 (요청한 모델의 학습이 끝나기를 기다렸다가) 채점."""
    from datetime import datetime

    from fab.bench.mlebench.remote import prepare_grading
    from fab.remote import settings, submit_item

    if not bench_comps(cid):
        sys.exit(f"config.yaml 에 mlebench: {cid} 인 대회가 없음")
    cfg = settings()
    prepare_grading(cfg, [cid])
    key = f"{cid}__{datetime.now():%Y%m%d-%H%M%S}"
    submit_item(cfg, "grade", key, dict(cid=cid, groups=groups, note=note))
    print(f"채점 요청: {key} ({len(groups)}개 묶음) → 학습이 끝난 뒤 fab remote kick, 결과는 fab sync")


def agents() -> pd.DataFrame:
    """공개 리더보드 에이전트의 실행별 점수 (runs/<run_group>/*.json) 중 TABULAR 대회 → results/mlebench_agents.csv.
    실험(experiment) = 하네스 × LLM × 자원 설정. 매핑은 runs/run_group_experiments.csv (README 의 실험 설명 표 참고)."""
    runs = MLEBENCH_DIR / "runs"
    mapping = runs / "run_group_experiments.csv"
    groups = {}
    for blob in resolve_lfs({0: mapping.read_text()})[0]:  # 충돌 양쪽을 합침 (뒤에 온 쪽이 이김)
        df = pd.read_csv(io.BytesIO(blob), skipinitialspace=True)
        groups |= dict(zip(df.run_group.str.strip(), df.experiment_id.str.strip()))
    reports = [p for p in sorted(runs.glob("*/*.json")) if p.parent.name in groups]
    rows = []
    for p, blobs in resolve_lfs({p: p.read_text() for p in reports}).items():
        for blob in blobs:
            for r in json.loads(blob)["competition_reports"]:
                if r["competition_id"] in TABULAR:
                    rows.append(dict(experiment=groups[p.parent.name], run_group=p.parent.name, report=p.name,
                                     **{k: r.get(k) for k in ["competition_id", "score", "valid_submission", "any_medal",
                                                              "gold_medal", "above_median", "created_at"]}))
    if not rows:  # 빈 표로 덮어쓰지 않음
        sys.exit(f"{runs} 에서 TABULAR 대회 실행을 못 찾음 (run_group {len(groups)}개, 보고서 {len(reports)}개)")
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(AGENTS, index=False)
    print(f"{AGENTS}: 실험 {df.experiment.nunique()}개, 실행 {len(df)}줄 ({', '.join(sorted(df.competition_id.unique()))})")
    return df


def medal(row: dict) -> str:
    for m in ("gold", "silver", "bronze"):
        if row.get(f"{m}_medal"):
            return m
    return "중앙값 위" if row.get("above_median") else "-"


def main(argv: list[str] | None = None):
    argv = sys.argv[2:] if argv is None else argv
    if argv[:1] == ["agent"]:  # 트랙 B: 에이전트 실행 (fab/agent/__init__.py)
        from fab.agent import main as agent_main

        return agent_main(argv[1:])
    ap = argparse.ArgumentParser(prog="fab mleb", description="MLE-bench 채점 (대회 준비는 fab download)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade", help="MLE-bench test 로 채점 요청 (원격 정리 작업이 학습이 끝나기를 기다려 채점) → fab sync 로 results/mlebench.csv")
    g.add_argument("target", help="MLE-bench 대회 id 또는 fab 대회 이름")
    g.add_argument("models", nargs="+", help="fab 모델 이름 (여러 개면 예측 평균)")
    g.add_argument("--each", action="store_true", help="모델마다 따로 채점 (평균하지 않음)")
    g.add_argument("--note", default="")
    sub.add_parser("agents", help="공개 리더보드 에이전트의 표 형식 대회 점수 → results/mlebench_agents.csv")
    sub.add_parser("agent", help="에이전트 직접 실행 (fab mleb agent -h)")
    args = ap.parse_args(argv)

    if args.cmd == "agents":
        agents()

    if args.cmd == "grade":
        cid = args.target
        if (COMP_DIR / cid / "config.yaml").exists() and load_competition(cid).mlebench:
            cid = load_competition(cid).mlebench
        request_grade(cid, [[m] for m in args.models] if args.each else [args.models], args.note)
