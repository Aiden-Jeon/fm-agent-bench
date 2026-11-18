"""fab/agent/lineage.py 의 단위 테스트 (가짜 실행 폴더로).

    uv run --frozen python -m unittest discover -s tests -v
"""

import io
import json
import shlex
import tarfile
import tempfile
import unittest
from pathlib import Path

from fab.agent import lineage as L

SUB = f"{L.HOME}/submission/submission.csv"


def make_run(d: Path, harness: str, files: dict[str, str], log_rows: list[dict] | None = None,
             submission_from: str = SUB, submission: str = "id,y\n0,1\n"):
    """가짜 회수 폴더: meta.json · 하네스 기록 · workspace.tgz (files 의 키는 묶음 안 이름, workspace/… 또는 home_submission/…)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(dict(key=d.name, harness=harness, submission_from=submission_from)))
    (d / "submission.csv").write_text(submission)
    if log_rows is not None:
        (d / f"{harness}.jsonl").write_text("\n".join(json.dumps(r) for r in log_rows) + "\n")
    with tarfile.open(d / "workspace.tgz", "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def codex(*commands, exit_codes=None):
    rows = [{"type": "thread.started"}]
    for i, c in enumerate(commands):
        code = (exit_codes or {}).get(i, 0)
        rows.append({"type": "item.completed", "item": {"id": f"item_{i}", "type": "command_execution",
                                                         "command": f"/bin/bash -lc {shlex.quote(c)}", "exit_code": code,
                                                         "aggregated_output": ""}})
    return rows


class CodeRefs(unittest.TestCase):
    def roles(self, code):
        return {(r.pattern, r.role) for r in L.code_refs(code)}

    def test_read_write_calls(self):
        got = self.roles("import numpy as np, pandas as pd\n"
                         "tr = pd.read_csv('/tmp/mleb_home/data/train.csv')\n"
                         "a = np.load('oof_lgb.npy')\n"
                         "np.save('feats.npy', a)\n"
                         "pd.DataFrame(a).to_csv('/tmp/mleb_home/submission/submission.csv', index=False)\n")
        self.assertEqual(got, {("/tmp/mleb_home/data/train.csv", "read"), ("oof_lgb.npy", "read"), ("feats.npy", "write"),
                               ("/tmp/mleb_home/submission/submission.csv", "write")})

    def test_paths_fstring_join_and_variables(self):
        got = self.roles("from pathlib import Path\nimport os, numpy as np, joblib\n"
                         "root = Path('results')\nD = '/tmp/mleb_work'\n"
                         "for seed in range(3):\n    np.savez(root / f'icl_s{seed}.npz', p=1)\n"
                         "m = joblib.load(os.path.join(D, 'model.pkl'))\n"
                         "out = Path('/tmp/mleb_home/submission/submission.csv')\n"
                         "df.to_csv(out.with_suffix('.tmp'))\nout.with_suffix('.tmp').replace(out)\n")
        self.assertIn(("results/icl_s*.npz", "write"), got)
        self.assertIn(("/tmp/mleb_work/model.pkl", "read"), got)
        self.assertIn(("/tmp/mleb_home/submission/submission.tmp", "write"), got)
        self.assertIn(("/tmp/mleb_home/submission/submission.csv", "write"), got)
        # 대입한 글자의 언급은 읽기·쓰기로 잡힌 경로와 겹치면 빠짐
        self.assertNotIn(("/tmp/mleb_home/submission/submission.csv", "mention"), got)

    def test_literal_loop_expands(self):
        got = self.roles("import numpy as np\nfor name in ['neural2', 'neurlong']:\n"
                         "    np.savez(f'{name}_combined.npz', a=1)\n")
        self.assertEqual(got, {("neural2_combined.npz", "write"), ("neurlong_combined.npz", "write")})

    def test_open_mode_copy_and_mentions(self):
        got = self.roles("import shutil, pickle\nwith open('cache.pkl', 'wb') as f: pickle.dump(1, f)\n"
                         "x = pickle.load(open('feat.pkl', 'rb'))\nshutil.copy('best.csv', 'final.csv')\n"
                         "files = ['pred_a.npy', 'pred_b.npy']\ns = 'a'.replace('x.npz', 'y.npz')\n")
        self.assertIn(("cache.pkl", "write"), got)
        self.assertIn(("feat.pkl", "read"), got)
        self.assertIn(("best.csv", "read"), got)
        self.assertIn(("final.csv", "write"), got)
        self.assertIn(("pred_a.npy", "mention"), got)  # 함수에 넘겨 역할을 모르는 경로
        self.assertNotIn(("y.npz", "write"), got)  # str.replace 는 복사도 파일 언급도 아님
        self.assertNotIn(("y.npz", "mention"), got)

    def test_syntax_error_and_imports(self):
        self.assertIsNone(L.code_refs("def ("))
        self.assertEqual(L.code_imports("import numpy as np, os.path\nfrom sklearn.model_selection import KFold\n"
                                        "from .util import f\n"), [".util", "numpy", "os", "sklearn"])


class Shell(unittest.TestCase):
    def test_heredoc_script_then_run(self):
        runs, events, _ = L.parse_shell("cat > feat.py <<'PY'\nprint(1)\nPY\npython feat.py > feat.log 2>&1",
                                        L.WORK, 1, True)
        self.assertEqual([(e.path, e.content) for e in events], [(f"{L.WORK}/feat.py", "print(1)")])
        self.assertEqual([(r.kind, r.path) for r in runs], [("script", f"{L.WORK}/feat.py")])
        self.assertLess(events[0].order, runs[0].order)  # 같은 명령 안에서도 만든 뒤 실행

    def test_inline_loop_wrappers_cd_cp_sed(self):
        cmd = ("cd sub; OMP_NUM_THREADS=1 python - <<'PY'\nimport numpy as np\nnp.save('a.npy', 1)\nPY\n"
               "for n in x y; do nice -n 5 timeout 60 python /tmp/mleb_work/$n.py & done\n"
               "cp best.csv /tmp/mleb_home/submission/\nsed -i 's/a/b/' x.py")
        runs, events, cwd = L.parse_shell(cmd + "\nVERSION=6 OMP_NUM_THREADS=2 python z.py", L.WORK, 3, True)
        self.assertEqual(runs[-1].env, {"VERSION": "6"})  # 판을 고르는 환경 변수는 남기고 스레드 수는 뺌
        self.assertEqual(cwd, f"{L.WORK}/sub")
        kinds = [(r.kind, r.path, r.src) for r in runs]
        self.assertEqual(kinds[0][0], "inline")
        self.assertEqual(runs[0].cwd, f"{L.WORK}/sub")
        self.assertIn(("script", f"{L.WORK}/x.py", None), kinds)
        self.assertIn(("script", f"{L.WORK}/y.py", None), kinds)
        self.assertIn(("copy", SUB.replace("submission.csv", "best.csv"), f"{L.WORK}/sub/best.csv"), kinds)
        self.assertIn((f"{L.WORK}/sub/x.py", None, "sed -i"), [(e.path, e.content, e.kind) for e in events])

    def test_exit_codes(self):
        self.assertTrue(L.exit_ok(0))
        self.assertIsNone(L.exit_ok(143))  # 시간 제한으로 끊김: 쓰기 후보로 남김
        self.assertFalse(L.exit_ok(1))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_codex_chain(self):
        feat = ("import numpy as np, pandas as pd\ntr = pd.read_csv('/tmp/mleb_home/data/train.csv')\n"
                "np.save('feats.npy', tr.values)\n")
        final = ("import numpy as np, pandas as pd\nX = np.load('/tmp/mleb_work/feats.npy')\n"
                 "pd.DataFrame(X).to_csv('/tmp/mleb_home/submission/submission.csv')\n")
        broken = "import pandas as pd\npd.DataFrame().to_csv('/tmp/mleb_home/submission/submission.csv')\n"
        run = self.d / "run"
        make_run(run, "codex", {"workspace/feat.py": feat, "workspace/final.py": final, "workspace/broken.py": broken,
                                "workspace/feats.npy": "x"},
                 codex(f"cat > feat.py <<'PY'\n{feat}PY\npython feat.py", "python final.py", "python broken.py",
                       exit_codes={2: 1}))
        res = L.Lineage(run).result()
        self.assertEqual(res["final"], f"{L.WORK}/final.py")  # exit 1 인 broken.py 는 제출을 쓴 것으로 안 셈
        nodes = {n["id"]: n for n in res["nodes"]}
        read = next(x for x in nodes[res["final"]]["reads"] if x["pattern"] == "workspace/feats.npy")
        self.assertEqual((read["status"], [p["node"] for p in read["producers"]]), ("found", [f"{L.WORK}/feat.py"]))
        self.assertEqual(nodes[f"{L.WORK}/feat.py"]["version"], "log")  # heredoc 에 내용이 남음
        self.assertEqual(nodes[f"{L.WORK}/final.py"]["version"], "workspace")
        self.assertEqual(res["unresolved"], [])

    def test_ambiguous_and_not_found_are_left_open(self):
        a = "import numpy as np\nnp.save(f'{k}_p.npy', 1)\n"
        final = ("import numpy as np, pandas as pd\nx = np.load('m_p.npy') + np.load('gone.npy')\n"
                 "pd.DataFrame(x).to_csv('/tmp/mleb_home/submission/submission.csv')\n")
        run = self.d / "run"
        make_run(run, "codex", {"workspace/a.py": a, "workspace/b.py": a, "workspace/final.py": final,
                                "workspace/m_p.npy": "x"},
                 codex("python a.py", "python b.py", "python final.py"))
        res = L.Lineage(run).result()
        reads = {x["pattern"]: x for x in res["nodes"][0]["reads"]}
        self.assertEqual(reads["workspace/m_p.npy"]["status"], "ambiguous")  # 같은 구체성의 쓰기가 두 코드
        self.assertEqual(reads["workspace/gone.npy"]["status"], "not_found")
        self.assertEqual(len(res["unresolved"]), 2)
        self.assertEqual(len(res["nodes"]), 1)  # 고르지 않은 후보는 따라가지 않음

    def test_read_modify_write_follows_earlier_writer(self):
        base = "import pandas as pd\npd.DataFrame({'y': [1]}).to_csv('/tmp/mleb_home/submission/submission.csv')\n"
        fix = ("import pandas as pd\np = '/tmp/mleb_home/submission/submission.csv'; s = pd.read_csv(p); "
               "s.clip(0, 1).to_csv(p)\n")  # 한 줄에 읽기·쓰기 (Codex 인라인 꼴): 열 순서로 가름
        tmp = ("import pandas as pd, os\npd.DataFrame().to_csv('/tmp/mleb_home/submission/s.tmp.csv')\n"
               "os.replace('/tmp/mleb_home/submission/s.tmp.csv', '/tmp/mleb_home/submission/x.csv')\n")
        run = self.d / "run"
        make_run(run, "codex", {"workspace/base.py": base, "workspace/fix.py": fix, "workspace/tmp.py": tmp},
                 codex("python base.py", "python tmp.py", "python fix.py"))
        res = L.Lineage(run).result()
        self.assertEqual(res["final"], f"{L.WORK}/fix.py")
        read = res["nodes"][0]["reads"][0]
        self.assertTrue(read.get("rewrites"))  # 제출을 읽어 고쳐 다시 씀 → 앞서 쓴 base.py 로 이어 감
        self.assertEqual([p["node"] for p in read["producers"]], [f"{L.WORK}/base.py"])
        lin = L.Lineage(run)
        tmp_run = next(r for r in lin.runs if r.path == f"{L.WORK}/tmp.py")
        lin.trace(tmp_run)
        statuses = {x["pattern"]: x["status"] for x in lin.nodes[f"{L.WORK}/tmp.py"]["reads"]}
        # 쓴 뒤 검사로 다시 읽는 것 (AIDE 노드의 검사 서버 올리기) 은 앞 생산자로 이어 가지 않음
        check = ("import pandas as pd\npd.DataFrame().to_csv('/tmp/mleb_home/submission/submission.csv')\n"
                 "f = open('/tmp/mleb_home/submission/submission.csv', 'rb')\n")
        run2 = self.d / "run2"
        make_run(run2, "codex", {"workspace/base.py": base, "workspace/check.py": check},
                 codex("python base.py", "python check.py"))
        res2 = L.Lineage(run2).result()
        self.assertEqual((res2["final"], res2["nodes"][0]["reads"][0]["status"], len(res2["nodes"])),
                         (f"{L.WORK}/check.py", "self", 1))
        self.assertEqual(statuses["/tmp/mleb_home/submission/s.tmp.csv"], "self")  # 임시 파일은 그대로 self

    def test_exec_of_modified_script_counts_its_writes(self):
        compose = "import pandas as pd\nw = .3\npd.DataFrame({'y': [w]}).to_csv('/tmp/mleb_home/submission/submission.csv')\n"
        inline = ("from pathlib import Path\ns = Path('compose.py').read_text().replace('w = .3', 'w = .35')\n"
                  "exec(compile(s, 'v6', 'exec'))\n")
        run = self.d / "run"
        make_run(run, "codex", {"workspace/compose.py": compose},
                 codex("python compose.py", f"VERSION=6 python - <<'PY'\n{inline}PY"))
        res = L.Lineage(run).result()
        node = res["nodes"][0]
        self.assertEqual(node["kind"], "inline")  # 끌어온 코드를 돌린 인라인이 마지막 쓰기
        self.assertEqual((node["exec"], node.get("exec_text_modified")), (["workspace/compose.py"], True))
        self.assertEqual(node["executions"][0]["env"], {"VERSION": "6"})
        # 앞부분만 잘라 돌리면 (Is-this × Claude Code 의 r3.py) 끌어온 파일의 쓰기는 안 셈
        part = "exec(open('compose.py').read().split('pd.DataFrame')[0])\nprint(w)\n"
        run2 = self.d / "run2"
        make_run(run2, "codex", {"workspace/compose.py": compose, "workspace/part.py": part},
                 codex("python compose.py", "python part.py"))
        self.assertEqual(L.Lineage(run2).result()["final"], f"{L.WORK}/compose.py")

    def test_changed_after_run_and_installed_files(self):
        final = "import pandas as pd\npd.DataFrame().to_csv('/tmp/mleb_home/submission/submission.csv')\n"
        run = self.d / "run"
        record = "pkg/mod.py,sha256=x,1\npkg-1.0.dist-info/RECORD,,\n"
        make_run(run, "codex", {"workspace/final.py": final, "workspace/lib/pkg/mod.py": "import x",
                                "workspace/lib/pkg-1.0.dist-info/RECORD": record},
                 codex("python final.py", "sed -i 's/a/b/' final.py"))
        lin = L.Lineage(run)
        self.assertNotIn(f"{L.WORK}/lib/pkg/mod.py", lin.ws.members)  # pip 가 깐 파일은 뺌 (installed_files)
        node = lin.result()["nodes"][0]
        self.assertEqual((node["version"], node["changed_after_run"]), ("workspace", True))

    def test_claude_code_write_edit(self):
        code = "import pandas as pd\npd.read_csv('x.csv').to_csv('/tmp/mleb_home/submission/submission.csv')\n"
        rows = [{"type": "assistant", "message": {"content": [
                    {"type": "tool_use", "id": "t1", "name": "Write",
                     "input": {"file_path": "/tmp/mleb_work/s.py", "content": code}},
                    {"type": "tool_use", "id": "t2", "name": "Edit",
                     "input": {"file_path": "/tmp/mleb_work/s.py", "old_string": "x.csv", "new_string": "y.csv"}},
                    {"type": "tool_use", "id": "t3", "name": "Bash", "input": {"command": "cd /tmp/mleb_work && python s.py"}},
                    {"type": "tool_use", "id": "t4", "name": "Bash", "input": {"command": "python missing.py"}}]}},
                {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t4", "is_error": True}]}}]
        run = self.d / "run"
        make_run(run, "claude-code", {"workspace/s.py": code.replace("x.csv", "z.csv"), "workspace/y.csv": "a"}, rows)
        res = L.Lineage(run).result()
        node = res["nodes"][0]
        self.assertEqual((res["final"], node["version"]), (f"{L.WORK}/s.py", "log"))  # Write + Edit 을 적용한 판
        self.assertEqual([x["pattern"] for x in node["reads"]], ["workspace/y.csv"])  # 최종본 (z.csv) 이 아님

    def test_aide_working_file(self):
        first = "import numpy as np\nnp.save('./working/oof.npy', 1)\n"
        best = ("import numpy as np, pandas as pd\ntr = pd.read_csv('./input/train.csv')\n"
                "o = np.load('./working/oof.npy')\npd.DataFrame(o).to_csv('./submission/submission.csv')\n")
        journal = {"nodes": [
            {"id": "n0", "step": 0, "code": first, "parent": None, "is_buggy": False, "exc_type": None, "metric": {"value": 1}},
            {"id": "n1", "step": 1, "code": best, "parent": "n0", "is_buggy": False, "exc_type": None, "metric": {"value": 2}}]}
        run = self.d / "run"
        make_run(run, "aide", {"workspace/logs/exp/journal.json": json.dumps(journal),
                               "workspace/workspaces/exp/best_solution/node_id.txt": "n1",
                               "workspace/workspaces/exp/working/oof.npy": "x"},
                 None, submission_from=f"{L.AIDE_EXP}/best_submission/submission.csv")
        res = L.Lineage(run).result()
        self.assertEqual(res["final"], "aide:n1")
        node = next(n for n in res["nodes"] if n["id"] == "aide:n1")
        reads = {x["pattern"]: x for x in node["reads"]}
        self.assertEqual(reads["workspace/workspaces/exp/input/train.csv"]["status"], "data")
        self.assertEqual([p["node"] for p in reads["workspace/workspaces/exp/working/oof.npy"]["producers"]], ["aide:n0"])
        self.assertEqual([a["id"] for a in node["aide_ancestry"]], ["n1", "n0"])


if __name__ == "__main__":
    unittest.main()
