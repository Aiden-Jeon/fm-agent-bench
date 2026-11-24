"""competitions/fc-*/description.md 만들기 (에이전트에게 주는 설명, 영어). `fab fc describe` 가 부름. 만든 파일은 커밋하고
손으로 고쳐도 됨 (다시 만들면 덮어씀). 이름 · 출처 · 지역 · 기관은 안 씀 (docs/forecast.md 8 의 익명화). 열 뜻은 확인할 수
있는 정도까지만 (fev 데이터 카드에 열 정의가 없어 단위는 "original units").
"""

from __future__ import annotations

import json

import pandas as pd

FREQ = {"h": ("hourly", "hour"), "D": ("daily", "day"), "W-SUN": ("weekly", "week"), "ME": ("monthly", "month"),
        "MS": ("monthly", "month")}
_EVENT = "past only: operational event indicator"
# task → (제목, 무엇의 시계열인지, 공변량 뜻). 공변량 뜻의 머리말 static / known / past only 가 종류
TEXT = {
    "redset_1H": ("Forecast the hourly workload of database clusters",
                  "Each series is the hourly workload (an aggregate activity measure, in its original units) of one "
                  "database cluster of a cloud data warehouse service.",
                  {"subset": "static: how the cluster is deployed (`provisioned` or `serverless`), the same value on "
                             "every row of a series"}),
    "solar_with_weather_1H": ("Forecast the hourly output of a solar power installation from weather",
                              "The single series is the hourly energy output (in its original units) of one solar power "
                              "installation, together with hourly weather variables.",
                              {"wind_speed": "known: wind speed", "day_length": "known: length of the day (minutes)",
                               "humidity": "known: relative humidity (%)", "rain_1h": "known: rain in the hour",
                               "snow_1h": "known: snow in the hour", "temp": "known: air temperature (°C)",
                               "pressure": "known: air pressure (hPa)",
                               "global_horizontal_irradiance": "past only: global horizontal solar irradiance",
                               "clouds_all": "past only: cloud cover (%)"}),
    "ercot_1D": ("Forecast daily electricity load of grid regions",
                 "Each series is the daily electricity load (demand, in its original units) of one region of a power "
                 "grid.", {}),
    "epf_be": ("Forecast the hourly day-ahead electricity price",
               "The single series is the hourly day-ahead price of electricity in one wholesale market, with two "
               "day-ahead forecasts published before the price is set.",
               {"Generation forecast": "known: day-ahead forecast of generation",
                "System load forecast": "known: day-ahead forecast of the system load"}),
    "rohlik_orders_1D": ("Forecast daily orders at warehouses of an online grocery retailer",
                         "Each series is the daily number of customer orders handled by one warehouse of an online "
                         "grocery retailer.",
                         {"holiday": "known: public holiday indicator", "shops_closed": "known: shops closed indicator",
                          "winter_school_holidays": "known: winter school holiday indicator",
                          "school_holidays": "known: school holiday indicator", "shutdown": _EVENT,
                          "mini_shutdown": _EVENT, "blackout": _EVENT, "mov_change": "past only: operational measurement",
                          "frankfurt_shutdown": _EVENT, "precipitation": "past only: precipitation",
                          "snow": "past only: snow", "user_activity_1": "past only: user activity measure",
                          "user_activity_2": "past only: user activity measure"}),
    "solar_1D": ("Forecast daily output of solar power plants",
                 "Each series is the daily power output (in its original units) of one photovoltaic power plant.", {}),
    "ecdc_ili": ("Forecast weekly influenza-like illness rates",
                 "Each series is the weekly rate of influenza-like illness (consultations per population, in its original "
                 "units) reported by one region's surveillance system.", {}),
    "hospital": ("Forecast monthly patient demand for medical products",
                 "Each series is the monthly count of patient demand for one medical product.", {}),
    "us_consumption_1M": ("Forecast monthly consumer spending by category",
                          "Each series is the monthly consumer spending (from national accounts, in its original units) "
                          "for one spending category.", {}),
    "hospital_admissions_1D": ("Forecast daily hospital admissions",
                               "Each series is the daily number of patient admissions at one hospital.", {}),
}


def text(name: str, comp, info: dict) -> str:
    title, what, cols = TEXT[name]
    d = comp.data_dir
    tr = pd.read_csv(d / "train.csv", dtype={"id": str})
    te = pd.read_csv(d / "test.csv", dtype={"id": str})
    sm = pd.read_csv(d / "sample_submission.csv", dtype={"id": str})
    adj, unit = FREQ[info["freq"]]
    n, h, m = info["n_items"], info["horizon"], info["seasonality"]
    lens = tr.groupby("id").size()
    lens_s = f"{lens.min():,}" if lens.min() == lens.max() else \
        f"{lens.min():,} to {lens.max():,} (median {int(lens.median()):,})"
    if n == 1:
        horizon = f"You get the full {adj} history of the series and must forecast the next {h} {unit}s (the forecast horizon)."
    elif te.groupby("id").timestamp.min().nunique() == 1:
        horizon = (f"For every series you get its full {adj} history up to the same forecast start and must forecast the "
                   f"next {h} {unit}s (the forecast horizon).")
    else:  # fev 의 정수 cutoff: 시리즈마다 자기 끝의 horizon 개 (docs/forecast.md 3)
        horizon = (f"For every series you get its full {adj} history and must forecast the next {h} {unit}s after the end "
                   f"of that history (the forecast horizon). Series end at different times, so the forecast horizons of "
                   f"different series are not all the same calendar period.")
    known = [c for c in te.columns if c not in ("id", "timestamp")]
    static = [c for c in cols if cols[c].startswith("static")]
    past_only = [c for c in tr.columns if c not in ("id", "timestamp", "target", *known, *static)]
    n_cov = len(tr.columns) - 3
    covs = f" and {n_cov} covariate{'s' if n_cov > 1 else ''}" if n_cov else ""
    miss_x = {c: int(tr[c].isna().sum()) for c in tr.columns[3:] if tr[c].isna().any()}
    lines = [
        f"# {title}", "", "## Description", "",
        f"{what} There {'is' if n == 1 else 'are'} {n} series. {horizon}", "",
        "Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, "
        "…, 0.9 quantiles of the predictive distribution.", "",
        "The `id` values are random identifiers that carry no information beyond grouping the rows of one series.", "",
        "## Evaluation", "",
        "Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series "
        "and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and "
        "each series is scaled by its in-sample seasonal error: the mean absolute difference "
        f"y_t − y_{{t−{m}}} over that series' history in `train.csv` (seasonal period {m}). "
        "Series whose seasonal error is zero or undefined do not count.", "",
        "The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a "
        "secondary metric.", "",
        "## Submission File", "",
        f"One row per (`id`, `timestamp`) in `test.csv` ({len(te):,} rows): the point forecast and the 9 quantiles. "
        "The file should contain a header and have the following format:", "",
        "```", ",".join(sm.columns), *[",".join(map(str, r)) for r in sm.head(2).itertuples(index=False)], "...", "```", "",
        "## Dataset Description", "",
        f"- `train.csv` — the history: `id`, `timestamp`, `target`{covs}. {len(tr):,} rows, {lens_s} {unit}s per series, "
        f"from {tr.timestamp.min()[:10]} to {tr.timestamp.max()[:10]}",
        "- `test.csv` — the rows to forecast: `id`, `timestamp`" + (" and the known covariates" if known else "")
        + f", {h} {unit}s per series from {te.timestamp.min()[:10]} to {te.timestamp.max()[:10]}",
        "- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)",
        "", "Columns", "",
        "- `id` — series identifier", f"- `timestamp` — time of the observation ({adj}, regular within each series)",
        "- `target` — the value to forecast",
        *[f"- `{c}` — {cols[c]}" for c in [*static, *known, *past_only]],
    ]
    if known or past_only:
        lines += ["", "\"known\" covariates are given for both the history and the forecast horizon (in `test.csv`). "
                      "\"past only\" covariates are given in `train.csv` only; their future values are not available."]
    tail = "; covariates: " + ", ".join(f"`{k}` {v:,}" for k, v in miss_x.items()) if miss_x else ""
    lines += ["", f"Missing values: {int(tr.target.isna().sum()):,} in `target`{tail}."]
    return "\n".join(lines) + "\n"


def write(name: str):
    from fab.bench.forecast import ANSWERS, make_comp

    comp = make_comp(name)
    info = json.loads((ANSWERS / f"{comp.name}.json").read_text())
    (comp.dir / "description.md").write_text(text(name, comp, info))
    print(f"{comp.dir / 'description.md'}")
