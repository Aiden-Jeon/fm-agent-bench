"""fab/bench/forecast 의 지표 · 제출 채점과 elo() 의 기준점 인자 단위 테스트 (네트워크 · fev 없이).

    uv run --frozen python -m unittest discover -s tests -v

fev 와 같은 값인지는 실제 데이터로 `fab fc check all` 이 봄 (옆 가상환경의 fev 로 채점해 비교).
"""

import unittest

import numpy as np
import pandas as pd

from fab.bench.forecast import metrics, score_frame
from fab.bench.tabarena import elo

LEVELS = [0.1, 0.5, 0.9]


class Metrics(unittest.TestCase):
    def test_seasonal_error(self):
        s = metrics.seasonal_error([np.array([1.0, 2, 4, 7]), np.array([5.0]), np.array([1.0, np.nan, 3])], m=1)
        self.assertAlmostEqual(s[0], (1 + 2 + 3) / 3)
        self.assertTrue(np.isnan(s[1]))  # 과거가 m 이하
        self.assertTrue(np.isnan(s[2]))  # 차가 모두 NaN

    def test_scores_by_hand(self):
        y = np.array([[10.0, 12.0]])
        point = np.array([[11.0, 11.0]])
        q = np.array([[[9.0, 11.0, 13.0], [9.0, 11.0, 13.0]]])
        s = np.array([2.0])
        r = metrics.scores(y, point, q, LEVELS, s)
        ql = [[2 * abs((10 - 9) * (0 - 0.1)), 2 * abs((10 - 11) * (1 - 0.5)), 2 * abs((10 - 13) * (1 - 0.9))],
              [2 * abs((12 - 9) * (0 - 0.1)), 2 * abs((12 - 11) * (0 - 0.5)), 2 * abs((12 - 13) * (1 - 0.9))]]
        self.assertAlmostEqual(r["SQL"], np.mean(ql) / 2)
        self.assertAlmostEqual(r["MASE"], 1 / 2)
        self.assertAlmostEqual(r["WAPE"], 1 / 11)
        self.assertAlmostEqual(r["WQL"], np.mean(ql) / 11)

    def test_undefined_scale_dropped(self):
        # 두 번째 시리즈는 계절 오차가 0 → SQL · MASE 에서 빠짐 (fev 의 _safemean)
        y = np.array([[10.0], [5.0]])
        q = np.repeat(y[..., None], 3, axis=2) + 1
        a = metrics.scores(y, y + 1, q, LEVELS, np.array([1.0, 0.0]))
        b = metrics.scores(y[:1], y[:1] + 1, q[:1], LEVELS, np.array([1.0]))
        self.assertAlmostEqual(a["SQL"], b["SQL"])
        self.assertAlmostEqual(a["MASE"], b["MASE"])


def answers():
    ts = ["2024-01-01", "2024-01-02"]
    ans = pd.DataFrame({"id": ["s1", "s1", "s2", "s2"], "timestamp": ts * 2, "target": [1.0, 2.0, 3.0, 4.0],
                        "scale": [1.0, 1.0, 2.0, 2.0]})
    return ans, dict(quantile_levels=LEVELS, horizon=2)


def submission(ans):
    sub = ans[["id", "timestamp"]].assign(predictions=ans.target + 1)
    for q in LEVELS:
        sub[str(q)] = ans.target + q
    return sub


class Submission(unittest.TestCase):
    def test_valid_and_order_free(self):
        ans, info = answers()
        sub = submission(ans)
        r = score_frame(sub, ans, info)
        self.assertTrue(r["valid_submission"])
        self.assertEqual(r["metric_error"], r["SQL"])
        shuffled = sub.iloc[[3, 1, 0, 2]].assign(timestamp=lambda d: d.timestamp + " 00:00:00")
        self.assertAlmostEqual(score_frame(shuffled, ans, info)["SQL"], r["SQL"])

    def test_invalid(self):
        ans, info = answers()
        sub = submission(ans)
        bad = {"열": sub.drop(columns="0.5"), "행": sub.iloc[1:], "겹침": pd.concat([sub.iloc[1:], sub.iloc[[1]]]),
               "시각": sub.assign(timestamp="2023-01-01"), "글자": sub.assign(timestamp="x"),
               "NaN": sub.assign(predictions=np.nan), "숫자 아님": sub.assign(predictions="a")}
        for k, v in bad.items():
            with self.subTest(k):
                self.assertFalse(score_frame(v, ans, info)["valid_submission"])


class EloAnchor(unittest.TestCase):
    def long(self):
        rng = np.random.default_rng(0)
        rows = [dict(method=m, dataset=f"d{i}", split="s", metric_error=base + rng.normal(0, 0.1))
                for i in range(12) for m, base in [("RandomForest", 1.0), ("Seasonal Naive", 1.2), ("A", 0.8)]]
        return pd.DataFrame(rows)

    def test_default_anchor_is_random_forest(self):
        t = elo.elo(self.long(), rounds=0)
        self.assertEqual(t.loc["RandomForest", "elo"], 1000)
        self.assertTrue(t.equals(elo.elo(self.long(), rounds=0, anchor="RandomForest")))

    def test_other_anchor_shifts_only(self):
        a, b = elo.elo(self.long(), rounds=0), elo.elo(self.long(), rounds=0, anchor="Seasonal Naive")
        self.assertAlmostEqual(b.loc["Seasonal Naive", "elo"], 1000)
        diff = (b.elo - a.elo.reindex(b.index)).to_numpy()
        self.assertTrue(np.allclose(diff, diff[0]))  # 기준점은 숫자를 옮길 뿐 간격은 같음
        with self.assertRaises(ValueError):
            elo.elo(self.long(), rounds=0, anchor="없음")

    def test_fill_anchor(self):
        long = self.long()
        long = long[~((long.method == "A") & (long.dataset == "d0"))]
        f = elo.fill(long, ["A"], anchor="Seasonal Naive")
        row = f[(f.method == "A") & (f.dataset == "d0")].iloc[0]
        sn = long[(long.method == "Seasonal Naive") & (long.dataset == "d0")].metric_error.iloc[0]
        self.assertEqual(row.metric_error, sn)
        self.assertEqual(row.imputed, 1.0)


if __name__ == "__main__":
    unittest.main()
