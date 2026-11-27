"""fab/bench/anomaly/mix.py 의 분위 · 섞는 규칙 · 진단값 · 임계값 단위 테스트 (점수 파일 · 네트워크 · pyod 없이).

    uv run --frozen python -m unittest discover -s tests -v

받은 점수가 기존 결과표와 같은지는 실제 파일로 `fab ad mix check` 가 봄.
"""

import unittest
from unittest import mock

import numpy as np

from fab.bench.anomaly import mix


class Quantiles(unittest.TestCase):
    def test_batch_rank(self):
        q = mix.q_batch(np.array([3.0, 1.0, 2.0, 2.0]))
        np.testing.assert_allclose(q, [3.5 / 4, 0.5 / 4, 2.0 / 4, 2.0 / 4])  # 같은 점수는 평균 순위
        self.assertTrue(((q > 0) & (q < 1)).all())

    def test_ref_position(self):
        ref = np.array([0.0, 1.0, 2.0, 3.0])
        q = mix.q_ref(np.array([-1.0, 1.0, 1.5, 9.0]), ref)
        # (작은 개수 + 같은 개수 / 2 + 0.5) / (n + 1)
        np.testing.assert_allclose(q, [0.5 / 5, (1 + 0.5 + 0.5) / 5, 2.5 / 5, 4.5 / 5])

    def test_ref_is_rowwise(self):
        # train 기준: 한 행의 분위는 같이 채점하는 다른 test 행과 무관
        ref = np.random.default_rng(0).normal(size=200)
        s = np.random.default_rng(1).normal(size=50)
        np.testing.assert_array_equal(mix.q_ref(s, ref)[:10], mix.q_ref(s[:10], ref))
        self.assertFalse(np.array_equal(mix.q_batch(s)[:10], mix.q_batch(s[:10])))

    def test_ref_inf(self):
        # train 점수의 +inf 는 맨 위 순위: 유한한 test 점수는 그 아래
        q = mix.q_ref(np.array([5.0]), np.array([1.0, 2.0, np.inf]))
        np.testing.assert_allclose(q, [2.5 / 4])


class Rules(unittest.TestCase):
    def setUp(self):
        self.a, self.b = np.array([0.1, 0.9, 0.5]), np.array([0.7, 0.3, 0.5])

    def test_rules(self):
        np.testing.assert_allclose(mix.combine(self.a, self.b, "mean"), [0.4, 0.6, 0.5])
        np.testing.assert_allclose(mix.combine(self.a, self.b, "max"), [0.7, 0.9, 0.5])
        np.testing.assert_allclose(mix.combine(self.a, self.b, "min"), [0.1, 0.3, 0.5])
        np.testing.assert_allclose(mix.combine(self.a, self.b, "w25"), 0.25 * self.a + 0.75 * self.b)
        np.testing.assert_allclose(mix.combine(self.a, self.b, "w75"), 0.75 * self.a + 0.25 * self.b)
        np.testing.assert_allclose(mix.combine(self.a, self.b, "tail"), -np.log(1 - self.a) - np.log(1 - self.b))
        with self.assertRaises(ValueError):
            mix.combine(self.a, self.b, "median")

    def test_tail_between_mean_and_max(self):
        # 한쪽만 아주 높은 행 (0.99, 0.5) 대 둘 다 조금 높은 행 (0.8, 0.8): mean 은 뒤, max · tail 은 앞을 더 높게
        a, b = np.array([0.99, 0.8]), np.array([0.5, 0.8])
        self.assertLess(*mix.combine(a, b, "mean"))
        self.assertGreater(*mix.combine(a, b, "max"))
        self.assertGreater(*mix.combine(a, b, "tail"))

    def test_names(self):
        self.assertEqual(mix.mix_name("tabpfn_unsup", "iforest", "mean", "train"), "tabpfn_unsup&iforest:mean")
        self.assertEqual(mix.mix_name("tabpfn_unsup", "iforest", "max", "batch"), "tabpfn_unsup&iforest:max@batch")
        self.assertEqual(mix.pick_name("tabpfn_unsup", "iforest", "if10", "train"), "tabpfn_unsup/iforest:if10")


class Diagnostics(unittest.TestCase):
    def test_top_mean_range(self):
        q = mix.q_batch(np.arange(100.0))
        self.assertAlmostEqual(mix.top_mean(q, q, 10), 1.0)  # 같은 행을 고름
        self.assertAlmostEqual(mix.top_mean(q, q[::-1], 10), -1.0)  # 고른 행이 다른 쪽에서는 맨 아래
        rng = np.random.default_rng(0)
        self.assertLess(abs(mix.top_mean(mix.q_batch(rng.normal(size=20000)), mix.q_batch(rng.normal(size=20000)), 10)), 0.05)  # 무관

    def test_masked_cluster(self):
        # IForest 로는 맨 위인 한 무리 (5%) 를 TFM 은 보통 행으로 봄 → if10 이 낮음. TFM 이 같은 행을 고르면 높음
        rng = np.random.default_rng(0)
        n, m = 2000, 100
        s_if = np.r_[rng.normal(size=n - m), rng.normal(size=m) + 6]
        agree = np.r_[rng.normal(size=n - m), rng.normal(size=m) + 6]
        masked = rng.normal(size=n)
        self.assertGreater(mix.diag_values(agree, s_if)["if10"], mix.diag_values(masked, s_if)["if10"] + 0.3)
        self.assertGreater(mix.diag_values(agree, s_if)["rho"], mix.diag_values(masked, s_if)["rho"])

    def test_nan_rows_dropped_inf_kept(self):
        s_if = np.arange(50.0)
        s_tfm = np.arange(50.0)
        s_tfm[0], s_tfm[49] = np.nan, np.inf  # NaN 행은 빼고, +inf 는 맨 위 순위
        d = mix.diag_values(s_tfm, s_if, s_if)
        self.assertAlmostEqual(d["rho"], 1.0)
        self.assertAlmostEqual(d["if10"], 1.0)
        self.assertAlmostEqual(d["knn10"], 1.0)
        self.assertTrue(np.isnan(mix.diag_values(s_tfm, s_if)["knn10"]))  # KNN 점수가 없으면 빈칸


class Threshold(unittest.TestCase):
    def test_fit_tau(self):
        # 진단값이 낮은 두 단위에서는 lo (IForest) 가, 높은 두 단위에서는 hi (TFM) 가 나음 → 그 사이의 중간점
        d = np.array([0.1, 0.2, 0.6, 0.8])
        hi, lo = np.array([0.5, 0.5, 0.9, 0.9]), np.array([0.9, 0.9, 0.6, 0.6])
        tau, val = mix.fit_tau(d, hi, lo)
        self.assertAlmostEqual(tau, 0.4)
        self.assertAlmostEqual(val, 0.9)

    def test_fit_tau_ends(self):
        d = np.array([0.1, 0.5, 0.9])
        self.assertEqual(mix.fit_tau(d, np.ones(3), np.zeros(3))[0], -np.inf)  # 늘 hi
        self.assertEqual(mix.fit_tau(d, np.zeros(3), np.ones(3))[0], np.inf)  # 늘 lo

    def test_fit_tau_tie_takes_middle(self):
        # 어느 τ 든 같은 평균 → 후보 (−inf, 0.3, 0.7, inf) 의 가운데
        d = np.array([0.1, 0.5, 0.9])
        tau, _ = mix.fit_tau(d, np.full(3, 0.7), np.full(3, 0.7))
        self.assertAlmostEqual(tau, 0.7)


class TrainScores(unittest.TestCase):
    def test_fold_split_is_methods_split(self):
        # E3 의 정리 점수 (methods.train_scores) 와 같은 나누기
        a, b = mix.fold_split(101, 3)
        perm = np.random.default_rng([3, 101]).permutation(101)
        np.testing.assert_array_equal(a, perm[:50])
        np.testing.assert_array_equal(b, perm[50:])
        self.assertEqual(sorted([*a, *b]), list(range(101)))

    def test_classifier_cross_fit(self):
        # 분류기: 한쪽 반으로 fit → 다른 반을 점수 (자기 행으로 fit 한 모형의 점수가 아님)
        X = np.arange(40.0).reshape(20, 2)
        y = np.r_[np.ones(10, int), np.zeros(10, int)]
        calls = []

        def fake(name, X_fit, y_fit, X_q, seed):
            calls.append((len(X_fit), len(X_q)))
            self.assertFalse({tuple(r) for r in X_fit} & {tuple(r) for r in X_q})
            return X_q[:, 0]

        with mock.patch("fab.bench.anomaly.methods.clf_scores", fake):
            s, kind = mix.train_scores("lgbm_clf+la100", X, 1, y)
        self.assertEqual(kind, "2fold")
        self.assertEqual(calls, [(10, 10), (10, 10)])
        np.testing.assert_array_equal(s, X[:, 0])

    def test_classifier_needs_both_classes_per_fold(self):
        X = np.zeros((20, 2))
        with self.assertRaises(ValueError):
            mix.train_scores("lgbm_clf+la100", X, 1, np.zeros(20, int))
        with self.assertRaises(ValueError):
            mix.train_scores("lgbm_clf+la100", X, 1)

    def test_nonfinite_count(self):
        self.assertEqual(mix.nonfinite(np.array([1.0, np.inf, -np.inf, np.nan, np.nan])), dict(posinf=1, neginf=1, nan=2))


if __name__ == "__main__":
    unittest.main()
