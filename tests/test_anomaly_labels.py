"""이상탐지: 라벨 준 이상을 context 에서 빼는 변형 (+dropla<k>) 의 이름 · 표 나누기, 점수 +inf 처리, S3 대회의 la 라벨
(docs/anomaly.md "라벨이 조금 있을 때 (2)"). 데이터 (FAB_ANOMALY) 없이 도는 것만.

    uv run --frozen python -m unittest discover -s tests -v
"""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from fab.bench import anomaly as A
from fab.bench.anomaly import methods as M
from fab.bench.anomaly import supervised as S
from fab.core.competition import Competition


def comp(name="ad-x", condition=None):
    ad = dict(dataset="fault", adbench="12_fault", split="s1", seed=1)
    if condition:
        ad["condition"] = condition
    return Competition(name=name, task="anomaly", target="label", id_col="id", metric="auroc", cv={}, anomaly=ad)


class DropNames(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(M.drop_spec("tabpfn_unsup+dropla10"), 10)
        self.assertIsNone(M.drop_spec("tabpfn_unsup"))
        self.assertIsNone(M.drop_spec("tabpfn_clf+la10"))
        self.assertIsNone(M.la_spec("tabpfn_unsup+dropla10"))  # 분류기의 +la 와 섞이지 않음
        for m, b in [("tabpfn_unsup+dropla75", "tabpfn_unsup"), ("kumo_unsup+dropla10", "kumo_unsup"), ("iforest+dropla1", "iforest")]:
            self.assertEqual(M.base(m), b)

    def test_known(self):
        for m in ["tabpfn_unsup+dropla1", "kumo_unsup+dropla75", "iforest+dropla100", "lof+dropla50", "knn+dropla5"]:
            self.assertTrue(M.is_known(m), m)
        for m in ["tabpfn_unsup+dropla7", "iforest+dropla10+clean", "iforest+trim10+dropla10", "tabpfn_clf+dropla10",
                  "tabpfn_clf+la10+dropla10", "nope+dropla10", "tabpfn_unsup+la10"]:
            self.assertFalse(M.is_known(m), m)
        # 기존 이름은 그대로
        for m in ["iforest", "iforest+clean", "tabpfn_unsup+trim10x2", "tabpfn_unsup+iftrim20", "kumo_clf+la1", "fomo0d+clean"]:
            self.assertTrue(M.is_known(m), m)
        self.assertFalse(M.is_known("kumo_clf+la75"))  # la 75 는 빼기 변형에만

    def test_kind(self):
        m = "tabpfn_unsup+dropla10"
        self.assertTrue(M.is_variant(m) and M.uses_la(m) and M.is_gpu(m))
        self.assertFalse(M.is_supervised(m) or M.is_clean(m) or M.is_kumo(m))
        self.assertTrue(M.is_variant("iforest+dropla10") and not M.is_gpu("iforest+dropla10"))
        self.assertTrue(M.uses_la("lgbm_clf+la10") and not M.uses_la("iforest+clean") and not M.uses_la("tabpfn_unsup"))
        self.assertFalse(M.is_variant("tabpfn_unsup") or M.is_variant("iforest+clean"))  # 공식 풀의 판정은 그대로
        self.assertEqual(M.opts("iforest+dropla10"), "default context=dropla la=0.1")
        self.assertEqual(M.opts("tabpfn_unsup+dropla75"), "n_permutations=10 version=v2 context=dropla la=0.75")
        self.assertEqual(M.opts("tabpfn_unsup"), "n_permutations=10 version=v2")

    def test_tables(self):
        off, s3, e2 = comp(), comp("adc-s3-x", dict(exp="s3", train="local")), comp("adc-e2-x", dict(exp="e2", rate=1))
        self.assertEqual(A.table_of(off, "tabpfn_unsup+dropla10"), A.SUPERVISED)
        self.assertEqual(A.table_of(off, "iforest+dropla75"), A.SUPERVISED)
        self.assertEqual(A.table_of(off, "kumo_clf+la10"), A.SUPERVISED)
        self.assertEqual(A.table_of(s3, "tabpfn_unsup"), A.SUPERVISED)
        self.assertEqual(A.table_of(s3, "lgbm_clf+la10"), A.SUPERVISED)
        # 기존 행의 자리는 그대로
        self.assertEqual(A.table_of(off, "tabpfn_unsup"), A.GRADES)
        self.assertEqual(A.table_of(off, "iforest+clean"), A.GRADES)
        self.assertEqual(A.table_of(off, "fomo0d"), A.GRADES)
        self.assertEqual(A.table_of(off, "tabpfn_unsup+trim10"), A.CONDITIONS)
        self.assertEqual(A.table_of(off, "kumo_unsup"), A.CONDITIONS)
        self.assertEqual(A.table_of(e2, "tabpfn_unsup"), A.CONDITIONS)
        row = A.grade_row(off, "iforest+dropla10", dict(score=0.9, metric_error=0.1, aupr=0.5), "n", dict(run="r", opts="o"))
        self.assertEqual((row["exp"], row["cond"]), ("base", ""))  # 조건 표와 같은 열 (보고가 exp 로 거름)


class PosInf(unittest.TestCase):
    def test_finite_untouched(self):
        s = np.array([0.5, -3.0, 12.25])
        out, n = M.top_posinf(s)
        self.assertIs(out, s)  # 유한하면 받은 배열 그대로 → 기존 결과는 안 바뀜
        self.assertEqual(n, 0)

    def test_posinf_to_top(self):
        s = np.array([0.5, np.inf, 12.25, np.inf, -7.0])
        out, n = M.top_posinf(s)
        self.assertEqual(n, 2)
        np.testing.assert_array_equal(out, [0.5, 13.25, 12.25, 13.25, -7.0])
        self.assertTrue(np.isposinf(s[1]))  # 입력은 안 바꿈
        self.assertTrue((out[[1, 3]] > out[[0, 2, 4]].max()).all())

    def test_other_nonfinite_left(self):
        for bad in (np.nan, -np.inf):
            out, n = M.top_posinf(np.array([1.0, bad, 2.0]))
            self.assertEqual(n, 0)
            self.assertFalse(np.isfinite(out).all())  # scores() 의 검사에서 실패하게 그대로
        out, n = M.top_posinf(np.array([1.0, np.nan, np.inf]))
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(out[1]) and out[2] == 2.0)
        out, n = M.top_posinf(np.array([np.inf, np.inf]))  # 유한한 점수가 없으면 그대로
        self.assertEqual(n, 0)
        self.assertTrue(np.isposinf(out).all())


class S3Labels(unittest.TestCase):
    def test_nested_counts(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.3).astype(int)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(S, "TRAIN_LABELS", Path(d)):
            c = comp("adc-s3-local-fault-s1", dict(exp="s3", train="local"))
            pd.DataFrame({"id": rng.permutation(500), "label": y}).to_csv(Path(d) / f"{c.name}.csv", index=False)
            t = S.s3_la_labels(c, [1, 10, 100])
            again = S.s3_la_labels(c, [10])
            other = S.s3_la_labels(comp("adc-s3-cluster-fault-s1", dict(exp="s3", train="cluster")).__class__(
                **{**c.__dict__, "anomaly": {**c.anomaly, "condition": dict(exp="s3", train="cluster")}}), [10])
        n = int(y.sum())
        self.assertEqual([int(t[f"la{k}"].sum()) for k in (1, 10, 100)], [math.ceil(0.01 * n), math.ceil(0.1 * n), n])
        self.assertTrue((t.la100.to_numpy() == y).all())  # la 100% = train 의 이상 모두
        self.assertTrue((t.la1 <= t.la10).all() and (t.la10 <= t.la100).all())  # 작은 la 의 집합이 큰 la 에 포함
        self.assertTrue((again.la10 == t.la10).all())  # 고정 seed
        self.assertFalse((other.la10 == t.la10).all())  # train 종류마다 다른 순열


if __name__ == "__main__":
    unittest.main()
