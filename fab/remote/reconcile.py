"""정리 작업 본체 (원격, Databricks serverless CPU, 내 신원). entry.py 가 부름. 로컬에서는 돌리지 않음.

한 번 돌 때마다 Volume 의 state/ 에서 처리 안 된 항목을 차례로 처리:
  1. exclude  에이전트 실행을 결과에서 뺌 (mlebench_runs.csv 행 삭제)
  2. fm       끝난 FM 학습 실행 → 예측·OOF 를 outputs/ 로, results.csv 에 행 추가 (fold 실행은 3 에서)
  3. batch    --split-folds 묶음의 fold 실행이 다 끝나면 합침 (merge_folds, 메모리는 fold 수에 비례하지 않음)
  4. agent    끝난 에이전트 실행 → SP Volume 의 결과를 outputs/_mlebench/agents/<key>/ 로 복사 → 채점 → 사후 검사
  5. grade    FM 채점 요청 → 요청한 모델의 학습이 아직 돌고 있으면 기다림 → 제출 파일 → 채점 → mlebench.csv
              (bench=tabarena 요청은 split 대회마다 fab 지표로 → tabarena.csv, fab/bench/tabarena)
  6. recheck  mlebench_runs.csv 의 사후 검사 열 다시 계산
항목마다 결과표를 Volume 의 tables/ 에 쓴 뒤 <key>.done.json 을 남김 (중간에 죽으면 그 항목만 다음에 다시).
항목 처리 중 예외(SystemExit 포함)가 나면 그 항목이 결과표에 쓴 행을 되돌리고 (다시 처리할 때 행이 겹치지 않게),
<key>.attempts.json 에 세어 MAX_ATTEMPTS 번 실패하면 오류로 처리 끝냄 (같은 오류로 계속 돌지 않게).
스케줄 없음: `fab remote kick` 으로 돌릴 때만. 아직 끝나지 않은 실행·그 모델의 채점 요청은 다음 kick 때 다시 봄.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd

from fab.core.competition import COMP_DIR, load_competition
from fab.core.results import RESULTS, merge_results
from fab.remote import KINDS, TABLES, now

TERMINAL = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
MAX_ATTEMPTS = 3


class State:
    def __init__(self, volume: Path):
        self.root = volume / "state"

    def _dir(self, kind: str) -> Path:
        return self.root / kind

    def all(self, kind: str) -> dict[str, dict]:
        d = self._dir(kind)
        if not d.is_dir():
            return {}
        return {p.name.removesuffix(".json"): json.loads(p.read_text()) for p in sorted(d.glob("*.json"))
                if not p.name.endswith((".done.json", ".attempts.json", ".progress.json"))}

    def is_done(self, kind: str, key: str) -> bool:
        return (self._dir(kind) / f"{key}.done.json").exists()

    def pending(self, kind: str) -> dict[str, dict]:
        return {k: m for k, m in self.all(kind).items() if not self.is_done(kind, k)}

    def done(self, kind: str, key: str, **info):
        self._dir(kind).mkdir(parents=True, exist_ok=True)
        (self._dir(kind) / f"{key}.done.json").write_text(json.dumps(dict(info, at=now()), indent=2, default=str))

    def fail(self, kind: str, key: str, err: str):
        self._dir(kind).mkdir(parents=True, exist_ok=True)  # state/batch/ 처럼 아직 폴더가 없을 수 있음
        f = self._dir(kind) / f"{key}.attempts.json"
        n = (json.loads(f.read_text())["n"] if f.exists() else 0) + 1
        f.write_text(json.dumps(dict(n=n, last_error=err[-3000:], at=now()), indent=2))
        print(f"  {kind}/{key}: 실패 {n}/{MAX_ATTEMPTS}\n{err[-1500:]}")
        if n >= MAX_ATTEMPTS:
            self.done(kind, key, status="error", error=err[-3000:])

    def any_pending(self) -> bool:
        return any(self.pending(k) for k in KINDS)


class Tables:
    """결과표: 시작할 때 Volume 의 tables/ → 코드 묶음의 results/ (fab 코드가 RESULTS 에 씀), 항목마다 되돌려 씀."""

    def __init__(self, volume: Path):
        self.remote = volume / "tables"
        if not self.remote.is_dir():  # 빈 표로 시작하면 fab sync 가 로컬 기록을 덮어씀 → 멈춤
            raise SystemExit(f"{self.remote} 없음 → 로컬에서 fab remote deploy (결과표 초기값을 올림)")
        RESULTS.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            if (self.remote / name).exists():
                shutil.copyfile(self.remote / name, RESULTS / name)

    def begin(self):
        """항목 하나를 처리하기 전 표 상태를 기억 (rollback 용). 표는 작아서 내용째 들고 있음."""
        self.saved = {n: (RESULTS / n).read_bytes() if (RESULTS / n).exists() else None for n in TABLES}

    def rollback(self):
        """실패한 항목이 표에 쓴 행을 되돌림. 이미 Volume 에 썼을 수 있으므로 되돌린 표를 다시 씀."""
        for name, data in self.saved.items():
            if data is None:
                (RESULTS / name).unlink(missing_ok=True)
            else:
                (RESULTS / name).write_bytes(data)
        self.flush()

    def flush(self):
        for name in TABLES:
            if (RESULTS / name).exists():
                tmp = self.remote / f".{name}.tmp"
                shutil.copyfile(RESULTS / name, tmp)
                tmp.replace(self.remote / name)


def run_state(w, run_id) -> dict:
    """Jobs API 로 실행 상태. `databricks air run` 실행도 Jobs 일회성 실행(ai_runtime_task)이라 run_id 가 같음
    (2026-11-11 확인: 같은 id 를 air get = SUCCESS, jobs get-run = TERMINATED/SUCCESS)."""
    r = w.jobs.get_run(int(run_id))
    st = r.state
    return dict(life=st.life_cycle_state.value if st.life_cycle_state else None,
                result=st.result_state.value if st.result_state else None, message=st.state_message or "")


def link_data(volume: Path):
    """대회 데이터 (fab databricks 가 올린 Volume 의 data/<대회>/) → competitions/<대회>/data (fab 코드가 읽는 자리)."""
    for d in COMP_DIR.iterdir():
        if (src := volume / "data" / d.name).is_dir() and not (d / "data").exists():
            (d / "data").symlink_to(src)


def step(name: str, items: dict, fn, state: State, tables: Tables, kind: str):
    """항목마다 fn(key, manifest) 실행. 처리가 끝나면 fn 이 done 을 남기고, 아직이면 그냥 돌아옴 (다음에 다시).
    fab 코드 일부는 sys.exit 로 멈추므로 SystemExit 도 그 항목의 실패로 셈 (정리 작업 전체가 죽지 않게)."""
    for key, m in items.items():
        tables.begin()
        try:
            fn(key, m)
        except (Exception, SystemExit):
            tables.rollback()
            state.fail(kind, key, traceback.format_exc())
    print(f"[{name}] {len(items)}개 확인 ({time.strftime('%H:%M:%S')})", flush=True)


def main(volume: str, *_):  # 예전 entry.py 는 job id 도 넘김 → 무시
    from databricks.sdk import WorkspaceClient

    from fab.agent import AGENT_RUNS, TA_TABLE, TABLE as AGENT_TABLE, grade_run, recheck_table, recover_from_progress, \
        run_table
    from fab.bench.mlebench import bench_comps, grade as grade_fm
    from fab.bench.mlebench.remote import Grader
    from fab.bench.tabarena import grade as grade_ta
    from fab.bench.tabarena.remote import Grader as TaGrader
    from fab.runner.run import load_target, merge_folds

    V = Path(volume)
    w = WorkspaceClient()
    state, tables, grader, ta_grader = State(V), Tables(V), Grader(V), TaGrader(V)
    link_data(V)

    def exclude(key, m):
        for k in m["keys"]:
            state.done("agent", k, excluded=m["reason"])
        for table in (AGENT_TABLE, TA_TABLE):
            if table.exists():
                t = pd.read_csv(table)
                t[~t.key.isin(m["keys"])].to_csv(table, index=False)
        tables.flush()
        state.done("exclude", key, keys=m["keys"])
        print(f"  제외: {', '.join(m['keys'])} ({m['reason']})")

    def fm(key, m):
        st = run_state(w, m["run_id"])
        if st["life"] not in TERMINAL:
            return
        if m.get("code"):  # CPU 작업 코드 (Workspace). 끝났으니 지움
            try:
                w.workspace.delete(m["code"], recursive=True)
            except Exception as e:  # 이미 지워졌으면 그대로
                print(f"  {key}: Workspace 코드 삭제 실패 (무시): {e!r}")
        res = Path(m["out"]) / "results"
        install_failed = None  # 실행기가 설치에 실패한 패키지 (모델이 실패해도 작업은 SUCCESS 라 여기서 드러냄)
        if (f := Path(m["out"]) / "install_failed.json").exists():
            install_failed = json.loads(f.read_text())
            print(f"  {key}: 설치 실패 {install_failed}")
        if m.get("batch"):  # fold 실행: 묶음이 다 끝나면 batch 단계에서 합침
            state.done("fm", key, result=st["result"], has_results=res.is_dir())
            return
        if not (res / "results.csv").exists():
            print(f"  {key}: {st['result']} — 결과 없음 ({st['message'][:200]})")
            state.done("fm", key, status="no_results", result=st["result"], message=st["message"][:500],
                       install_failed=install_failed)
            return
        merge_results(res, "databricks-cpu" if m.get("cpu") else "databricks")
        tables.flush()
        state.done("fm", key, result=st["result"])

    step("exclude", state.pending("exclude"), exclude, state, tables, "exclude")
    step("fm", state.pending("fm"), fm, state, tables, "fm")

    # fold 묶음: 묶음의 fold 실행이 전부 처리되면 모델마다 합침
    batches: dict[str, list[dict]] = {}
    for m in state.all("fm").values():
        if m.get("batch"):
            batches.setdefault(m["batch"], []).append(m)

    def batch(key, ms):
        if not all(state.is_done("fm", m["key"]) for m in ms):
            return
        comp = load_competition(ms[0]["comp"])
        y, _ = load_target(comp)
        failed = {}
        for model in ms[0]["models"]:
            dirs = {m["fold"]: Path(m["out"]) / "results" / comp.name / model / "folds" / str(m["fold"]) for m in ms}
            if reason := merge_folds(comp, model, y, dirs, "databricks"):
                failed[model] = reason
                print(f"  {key} {model}: 합치지 못함 — {reason}")
        tables.flush()
        state.done("batch", key, failed=failed)

    step("batch", {k: ms for k, ms in batches.items() if not state.is_done("batch", k)}, batch, state, tables,
         "batch")

    def agent(key, m):
        st = run_state(w, m["run_id"])
        if st["life"] not in TERMINAL:
            return
        d, src = AGENT_RUNS / key, Path(m["out"])
        d.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, d, dirs_exist_ok=True)
        else:
            print(f"  {key}: {st['result']} — 결과 폴더 없음 ({st['message'][:200]})")
        if recover_from_progress(d):
            print(f"  {key}: 실행기가 끝까지 못 감 ({st['message'][:200]}) → progress/ 사본으로 회수"
                  + (" (제출 있음)" if (d / "submission.csv").exists() else " (제출 사본 없음)"))
        report = None
        if (d / "submission.csv").exists() and m.get("bench") == "tabarena":
            # TabArena: 대회 이름은 제출 기록에만 (cid = 익명 id). 형식이 틀린 제출은 오류가 아니라 무효 (valid_submission)
            report = ta_grader.grade(m["comp"], d / "submission.csv")
            if str(report.get("error", "")).startswith("정답 없음"):
                raise RuntimeError(report["error"])
        elif (d / "submission.csv").exists():
            report = grader.grade(m["cid"], d / "submission.csv")
            if report.get("error"):  # 채점 자체가 실패 → 다시 시도 (MAX_ATTEMPTS 까지)
                raise RuntimeError(f"채점 실패:\n{report['error']}")
        table = run_table(m)
        if table.exists():  # 예전 시도가 행을 쓰고 done 을 못 남겼으면 그 행을 지우고 다시 씀
            t = pd.read_csv(table)
            t[t.key != key].to_csv(table, index=False)
        row = grade_run(key, m, report)
        tables.flush()
        state.done("agent", key, result=st["result"], score=row.get("score"))
        print(f"  {key}: {st['result']} score={row.get('score')} timed_out={row.get('timed_out')} "
              f"tfm_code={row.get('tfm_in_code') or '-'} tfm_run={row.get('tfm_runtime') or '-'} tokens={row.get('input_tokens')}/{row.get('output_tokens')}")

    step("agent", state.pending("agent"), agent, state, tables, "agent")

    def unfinished() -> set[tuple[str, str]]:
        """아직 처리가 안 끝난 (대회, 모델): 학습 중이거나, 끝났지만 합치기에 실패해 다시 시도할 FM 실행·fold 묶음.
        채점 요청은 이 모델들을 기다림 (끝난 줄 알고 예전 예측·없는 예측으로 채점하지 않게)."""
        def comps(m):  # TabArena 묶음 작업은 split 대회 여럿 (comps)
            return m.get("comps") or [m["comp"]]

        out = {(c, x) for m in state.pending("fm").values() for c in comps(m) for x in m["models"]}
        return out | {(c, x) for k, ms in batches.items() if not state.is_done("batch", k)
                      for m in ms for c in comps(m) for x in m["models"]}

    def grade(key, m):
        if m.get("bench") == "tabarena":
            return grade_tabarena(key, m)
        need = {(c.name, x) for c in bench_comps(m["cid"]) for g in m["groups"] for x in g}
        if waiting := need & unfinished():
            print(f"  {key}: 처리가 안 끝난 모델 기다림 {sorted(waiting)}")
            return
        # 묶음마다 채점이 끝나면 진행 기록에 남김 → 실패해 다시 시도할 때 끝난 묶음은 건너뜀 (행이 겹치지 않게)
        prog = state.root / "grade" / f"{key}.progress.json"
        graded = json.loads(prog.read_text()) if prog.exists() else []
        failed = {}
        for g in m["groups"]:
            if "+".join(g) in graded:
                continue
            tables.begin()
            rows, errors = grade_fm(m["cid"], [g], m.get("note", ""), grader.grade)
            if errors:
                tables.rollback()
                failed |= errors
                continue
            tables.flush()
            graded.append("+".join(g))
            prog.write_text(json.dumps(graded))
        if failed:  # 다시 시도 (MAX_ATTEMPTS 까지). 마지막엔 done 에 오류로 남음
            raise RuntimeError("채점 못 한 묶음:\n" + "\n".join(f"{k}: {v}" for k, v in failed.items()))
        state.done("grade", key, graded=graded)

    def grade_tabarena(key, m):
        """split 대회 × 모델. 끝난 대회는 진행 기록에 남겨 다시 시도할 때 건너뜀. 예측이 없는 (학습 실패) 것·채점에
        실패한 것 (무효 제출, test.npy 의 NaN 등) 은 같은 입력이면 다시 해도 같으므로 그 (대회, 모델) 만 빼고 나머지는 씀
        (done 의 missing · failed). 그 밖의 예외 (Volume 읽기 등) 는 대회째 되돌리고 다시 시도 (MAX_ATTEMPTS)."""
        need = {(c, x) for c in m["comps"] for x in m["models"]}
        if waiting := need & unfinished():
            print(f"  {key}: 처리가 안 끝난 모델 기다림 {len(waiting)}개 (예: {sorted(waiting)[:3]})")
            return
        prog = state.root / "grade" / f"{key}.progress.json"
        graded = json.loads(prog.read_text()) if prog.exists() else []
        missing, failed = {}, {}
        for c in m["comps"]:
            if c in graded:
                continue
            tables.begin()
            try:
                rows, errors = grade_ta([c], m["models"], m.get("note", ""), ta_grader.grade)
            except BaseException:
                tables.rollback()
                raise
            missing |= {k: v for k, v in errors.items() if v == "test 예측 없음"}
            failed |= {k: v for k, v in errors.items() if v != "test 예측 없음"}
            tables.flush()
            graded.append(c)
            prog.write_text(json.dumps(graded))
        state.done("grade", key, graded=len(graded), missing=missing, failed=failed)
        print(f"  {key}: 대회 {len(graded)}개 채점" + (f", 예측 없음 {len(missing)}개" if missing else "")
              + (f", 채점 실패 {len(failed)}개" if failed else ""))
        for k, v in list(failed.items())[:10]:
            print(f"    {k}: {v}")

    step("grade", state.pending("grade"), grade, state, tables, "grade")

    def recheck(key, m):
        recheck_table()
        tables.flush()
        state.done("recheck", key)

    step("recheck", state.pending("recheck"), recheck, state, tables, "recheck")

    # 아직 끝나지 않은 실행·기다리는 채점 요청 (다음 kick 때 다시)
    left = [k for k in KINDS for _ in state.pending(k)] + [k for k in batches if not state.is_done("batch", k)]
    print(f"남은 항목 {len(left)}개" + (" → 끝난 뒤 다시 fab remote kick" if left else ""))
