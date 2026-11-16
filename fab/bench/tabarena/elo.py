"""TabArena 식 Elo: 우리 FM·Agent 를 TabArena 공개 결과와 한 풀에 넣고 RandomForest (기본) = 1000.

TabArena (`autogluon/tabarena` c9f57b2, bencheval `EloHelper`·`BenchmarkEvaluator.compute_elo`, 논문 D.1) 와 같은 계산:

- Bradley-Terry 최대우도 (로지스틱 회귀, C=1e6, Elo = 400·계수 + 1000). 짝 비교는 같은 (데이터셋, split) 안에서,
  오차가 낮으면 이김, 같으면 반씩. 짝마다 가중치 1 / (그 데이터셋의 split 수) → 데이터셋마다 합 1
- 못 돈 (방법, 데이터셋, split) 은 RandomForest 기본 결과로 채움 (TabArena 의 fillna, `imputed` 열에 채운 비율)
- 95% 구간: 데이터셋을 복원 추출하는 부트스트랩 200번 (seed 0), 보정 전 값의 2.5%·97.5% 분위수 − 점추정.
  점추정은 부트스트랩 중앙값이 아니라 전체 데이터의 최대우도. 마지막에 RandomForest 가 1000 이 되게 옮김 (post_calibrate)

풀 (참가자 묶음) 이 다르면 Elo 도 다름: TabArena 리더보드는 방법이 훨씬 많아 같은 방법이라도 숫자가 다를 수 있음.

- fm: 51개 데이터셋의 모든 split (816). TabArena 기본 설정 7개 + 우리 FM (refit)
- agent: Agent 를 돌린 데이터셋의 r0f0. Agent 는 설정마다 회차 평균 (TabArena 의 seed 평균), 위 + 우리 GBDT (refit)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ANCHOR = "RandomForest"
SCALE, INIT, ROUNDS, SEED = 400, 1000.0, 200, 0
MIN_CI_DATASETS = 10  # 이보다 데이터셋이 적은 풀은 95% 구간을 비움 (fab ta elo)
# 우리 FM 의 51개 이름. kumo 는 범주 열이 있는 34개는 kumo_cat_refit, 없는 17개는 같은 설정인 kumo_c8k_refit (docs/tabarena.md "FM 프로토콜과 이름")
FM_MODELS = {"tabicl (ours)": ["tabicl_kv_refit"], "kumo (ours)": ["kumo_cat_refit", "kumo_c8k_refit"],
             "tabpfn_v2 (ours)": ["tabpfn_v2_refit"]}
GBDT_MODELS = {"lgbm (ours)": ["lgbm_refit"], "catboost (ours)": ["catboost_refit"]}


def _fit(err: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """err: (방법, 단위) 오차 행렬 (빈칸 없음), weight: 단위마다 가중치 → 보정 전 Elo (방법마다)."""
    m = err.shape[0]
    i, j = np.triu_indices(m, 1)
    a, b = err[i], err[j]  # (짝, 단위)
    win_i = ((a < b) + 0.5 * (a == b)) @ weight
    win_j = ((a > b) + 0.5 * (a == b)) @ weight
    x = np.zeros((len(i), m))
    x[np.arange(len(i)), i], x[np.arange(len(i)), j] = np.log(10), -np.log(10)
    keep = (win_i + win_j) > 0
    X = np.vstack([x[keep], x[keep]])
    y = np.r_[np.ones(keep.sum()), np.zeros(keep.sum())]
    w = np.r_[win_i[keep], win_j[keep]]
    lr = LogisticRegression(fit_intercept=False, C=1e6, tol=1e-10, max_iter=1000).fit(X, y, sample_weight=w)
    return SCALE * lr.coef_[0] + INIT


def elo(long: pd.DataFrame, rounds: int = ROUNDS, seed: int = SEED) -> pd.DataFrame:
    """long: method · dataset · split · metric_error (+ imputed). 빈칸은 미리 채워 둘 것 → 방법마다 elo · elo+ · elo- ·
    imputed (채운 단위 비율) · 데이터셋 수 · 평균 순위·승률 (TabArena 와 같은 단위 가중치)."""
    wide = long.pivot_table(index="method", columns=["dataset", "split"], values="metric_error", aggfunc="first")
    if wide.isna().any().any():
        raise ValueError(f"빈칸 {int(wide.isna().sum().sum())}개: 먼저 {ANCHOR} 로 채울 것")
    imp = long.pivot_table(index="method", columns=["dataset", "split"], values="imputed", aggfunc="first")
    ds = wide.columns.get_level_values("dataset").to_numpy()
    names, ds_idx = np.unique(ds, return_inverse=True)
    n_split = np.bincount(ds_idx)
    weight = 1.0 / n_split[ds_idx]
    err = wide.to_numpy(float)
    raw = _fit(err, weight)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(rounds):
        pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))  # 데이터셋별 뽑힌 횟수
        w = weight * pick[ds_idx]
        boots.append(_fit(err[:, w > 0], w[w > 0]))
    boots = np.array(boots)
    lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)

    anchor = list(wide.index).index(ANCHOR)
    rank = wide.rank(axis=0, method="average").to_numpy()
    m = len(wide)
    out = pd.DataFrame(dict(
        elo=raw - raw[anchor] + INIT,
        **{"elo+": np.clip(hi - raw, 0, None), "elo-": np.clip(raw - lo, 0, None)},
        rank=(rank * weight).sum(1) / weight.sum(),
        winrate=((m - rank) / (m - 1) * weight).sum(1) / weight.sum(),
        imputed=(imp.fillna(False).astype(bool).to_numpy() * weight).sum(1) / weight.sum(),
        datasets=len(names),
    ), index=wide.index)
    return out.sort_values("elo", ascending=False)


def fill(long: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """methods 마다 RandomForest 가 있는 모든 (데이터셋, split) 을 채움. 없는 곳은 RandomForest 값, imputed=True."""
    rf = long[long.method == ANCHOR][["dataset", "split", "metric_error"]]
    parts = []
    for m in methods:
        g = rf.merge(long[long.method == m][["dataset", "split", "metric_error"]], on=["dataset", "split"],
                     how="left", suffixes=("_rf", ""))
        g["imputed"] = g.metric_error.isna()
        g["metric_error"] = g.metric_error.fillna(g.metric_error_rf)
        parts.append(g.drop(columns="metric_error_rf").assign(method=m))
    return pd.concat(parts, ignore_index=True)


def _ref(ref: pd.DataFrame) -> pd.DataFrame:
    """TabArena 공개 결과 → method = TabArena 이름 (ta_name. 원래 method 열은 설정 이름이라 버림)."""
    return ref[["ta_name", "dataset", "split", "metric_error"]].rename(columns={"ta_name": "method"})


def _ours(grades: pd.DataFrame, entrants: dict[str, list[str]], splits: set | None = None) -> pd.DataFrame:
    """우리 채점 → method 이름 하나로. 이름이 여럿이면 (dataset, split) 마다 앞 이름 우선."""
    g = grades.sort_values("timestamp").drop_duplicates(["comp", "model"], keep="last")
    if splits is not None:
        g = g[g.split.isin(splits)]
    parts = []
    for name, models in entrants.items():
        rows = pd.concat([g[g.model == m].assign(_p=k) for k, m in enumerate(models)])
        rows = rows.sort_values("_p").drop_duplicates(["dataset", "split"])
        parts.append(rows[["dataset", "split", "metric_error"]].assign(method=name))
    return pd.concat(parts, ignore_index=True)


def fm_pool(ref: pd.DataFrame, grades: pd.DataFrame) -> pd.DataFrame:
    """51개 × 모든 split: TabArena 기본 설정 7개 + 우리 FM."""
    long = pd.concat([_ref(ref),
                      _ours(grades, FM_MODELS)], ignore_index=True)
    return fill(long, list(dict.fromkeys(long.method)))


def agent_pool(ref: pd.DataFrame, grades: pd.DataFrame, runs: pd.DataFrame, dataset_of: dict[str, str],
               split: str = "r0f0") -> pd.DataFrame:
    """Agent 를 돌린 데이터셋의 r0f0: TabArena 기본 설정 7개 + 우리 FM·GBDT + Agent (설정마다 회차 평균)."""
    runs = runs[runs.valid_submission.astype(bool) & runs.metric_error.notna()]
    agents = (runs.assign(dataset=runs.comp.map(dataset_of), split=split)
              .groupby(["config", "dataset", "split"], as_index=False).metric_error.mean()
              .rename(columns={"config": "method"}))
    agents["method"] = "agent: " + agents.method
    keep = set(agents.dataset)
    long = pd.concat([_ref(ref),
                      _ours(grades, {**FM_MODELS, **GBDT_MODELS}, {split}), agents], ignore_index=True)
    long = long[long.dataset.isin(keep) & (long.split == split)]
    return fill(long, list(dict.fromkeys(long.method)))


def agent_counts(runs: pd.DataFrame) -> pd.Series:
    """설정마다 평균에 든 회차 수 (표에 같이 적음)."""
    runs = runs[runs.valid_submission.astype(bool) & runs.metric_error.notna()]
    return runs.groupby("config").size().rename(lambda c: f"agent: {c}")
