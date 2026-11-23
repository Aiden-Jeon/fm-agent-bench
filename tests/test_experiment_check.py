"""fab/agent/__init__.py 의 제출 전 experiment 점검 (check_experiment) 단위 테스트 (dbx 를 가짜로).

    uv run --frozen python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import subprocess
import unittest
from unittest import mock

import fab.agent as A

SP = "00000000-1111-2222-3333-444444444444"
HOME = f"/Users/{SP}"


def cfg(experiment="fab-agent"):
    return dict(profile="test-profile", run_as=SP, experiment=experiment)


class FakeDbx:
    """workspace get-status 만 흉내: paths {경로: object_type}, 그 밖의 경로는 없음. denied 면 모두 권한 오류."""

    def __init__(self, paths=None, denied=False):
        self.paths, self.denied, self.calls = paths or {}, denied, []

    def __call__(self, c, *args, check=True, timeout=1200, **kw):
        assert args[:2] == ("workspace", "get-status"), args
        path = args[2]
        self.calls.append(path)
        if self.denied:
            return subprocess.CompletedProcess(args, 1, "", "Error: PERMISSION_DENIED: User does not have permission.\n")
        if path in self.paths:
            info = dict(object_id=123, object_type=self.paths[path], path=path)
            return subprocess.CompletedProcess(args, 0, json.dumps(info), "")
        return subprocess.CompletedProcess(args, 1, "", f"Error: Path ({path}) doesn't exist.\n")


class CheckExperiment(unittest.TestCase):
    def setUp(self):
        A._EXPERIMENT_OK.clear()

    def run_check(self, fake, c):
        out = io.StringIO()
        with mock.patch.object(A, "dbx", fake), contextlib.redirect_stdout(out):
            A.check_experiment(c)
        return out.getvalue()

    def stop_message(self, fake, c):
        with mock.patch.object(A, "dbx", fake), self.assertRaises(SystemExit) as e:
            A.check_experiment(c)
        return str(e.exception.code)

    def test_relative_name_under_sp_home(self):
        self.assertEqual(A.experiment_path(cfg()), f"{HOME}/fab-agent")

    def test_exists_passes_and_checks_once(self):
        fake = FakeDbx({HOME: "DIRECTORY", f"{HOME}/fab-agent": "MLFLOW_EXPERIMENT"})
        self.assertEqual(self.run_check(fake, cfg()), "")
        self.run_check(fake, cfg())  # 같은 프로세스에서 두 번째 설정
        self.assertEqual(fake.calls, [f"{HOME}/fab-agent"])

    def test_missing_home_stops_with_fix_commands(self):
        msg = self.stop_message(FakeDbx(), cfg())
        self.assertIn(f"{HOME}/fab-agent 없음", msg)
        self.assertIn("SP 홈 폴더", msg)
        self.assertIn(f"databricks workspace mkdirs {HOME} -p test-profile", msg)
        self.assertIn("permissions update directories <object_id>", msg)
        self.assertIn(f"experiments create-experiment {HOME}/fab-agent -p test-profile", msg)
        self.assertIn("experiments update-permissions <experiment_id>", msg)
        self.assertIn(f'"service_principal_name": "{SP}"', msg)
        self.assertIn("CAN_MANAGE", msg)
        self.assertIn("INTERNAL_ERROR", msg)

    def test_missing_experiment_with_home_fills_object_id(self):
        msg = self.stop_message(FakeDbx({HOME: "DIRECTORY"}), cfg())
        self.assertNotIn("mkdirs", msg)
        self.assertNotIn("SP 홈 폴더", msg)
        self.assertIn("permissions update directories 123 ", msg)
        self.assertIn("create-experiment", msg)

    def test_permission_error_warns_and_passes(self):
        out = self.run_check(FakeDbx(denied=True), cfg())
        self.assertIn("[경고]", out)
        self.assertIn("PERMISSION_DENIED", out)
        self.assertNotIn((cfg()["profile"], f"{HOME}/fab-agent"), A._EXPERIMENT_OK)

    def test_timeout_warns_and_passes(self):
        def slow(*a, **kw):
            raise subprocess.TimeoutExpired("databricks", 120)

        self.assertIn("[경고]", self.run_check(slow, cfg()))

    def test_absolute_experiment_path(self):
        path = "/Shared/fab/agent-exp"
        c = cfg(path)
        self.assertEqual(A.experiment_path(c), path)
        fake = FakeDbx({path: "MLFLOW_EXPERIMENT"})
        self.assertEqual(self.run_check(fake, c), "")
        self.assertEqual(fake.calls, [path])

    def test_absolute_missing_does_not_touch_sp_home(self):
        msg = self.stop_message(FakeDbx({"/Shared/fab": "DIRECTORY"}), cfg("/Shared/fab/agent-exp"))
        self.assertIn("create-experiment /Shared/fab/agent-exp", msg)
        self.assertNotIn("mkdirs", msg)
        self.assertNotIn("permissions update directories", msg)

    def test_path_is_not_experiment(self):
        msg = self.stop_message(FakeDbx({f"{HOME}/fab-agent": "DIRECTORY"}), cfg())
        self.assertIn("MLflow experiment 가 아님", msg)


if __name__ == "__main__":
    unittest.main()
