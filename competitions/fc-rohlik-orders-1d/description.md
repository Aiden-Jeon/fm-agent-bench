# Forecast daily orders at warehouses of an online grocery retailer

## Description

Each series is the daily number of customer orders handled by one warehouse of an online grocery retailer. There are 7 series. For every series you get its full daily history up to the same forecast start and must forecast the next 61 days (the forecast horizon).

Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, …, 0.9 quantiles of the predictive distribution.

The `id` values are random identifiers that carry no information beyond grouping the rows of one series.

## Evaluation

Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and each series is scaled by its in-sample seasonal error: the mean absolute difference y_t − y_{t−7} over that series' history in `train.csv` (seasonal period 7). Series whose seasonal error is zero or undefined do not count.

The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a secondary metric.

## Submission File

One row per (`id`, `timestamp`) in `test.csv` (427 rows): the point forecast and the 9 quantiles. The file should contain a header and have the following format:

```
id,timestamp,predictions,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
s001,2024-01-01,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0
s001,2024-01-02,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0,2272.0
...
```

## Dataset Description

- `train.csv` — the history: `id`, `timestamp`, `target` and 13 covariates. 7,186 rows, 682 to 1,122 (median 1,122) days per series, from 2020-12-05 to 2023-12-31
- `test.csv` — the rows to forecast: `id`, `timestamp` and the known covariates, 61 days per series from 2024-01-01 to 2024-03-01
- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)

Columns

- `id` — series identifier
- `timestamp` — time of the observation (daily, regular within each series)
- `target` — the value to forecast
- `holiday` — known: public holiday indicator
- `school_holidays` — known: school holiday indicator
- `shops_closed` — known: shops closed indicator
- `winter_school_holidays` — known: winter school holiday indicator
- `blackout` — past only: operational event indicator
- `frankfurt_shutdown` — past only: operational event indicator
- `mini_shutdown` — past only: operational event indicator
- `mov_change` — past only: operational measurement
- `precipitation` — past only: precipitation
- `shutdown` — past only: operational event indicator
- `snow` — past only: snow
- `user_activity_1` — past only: user activity measure
- `user_activity_2` — past only: user activity measure

"known" covariates are given for both the history and the forecast horizon (in `test.csv`). "past only" covariates are given in `train.csv` only; their future values are not available.

Missing values: 346 in `target`; covariates: `blackout` 346, `frankfurt_shutdown` 346, `mini_shutdown` 346, `mov_change` 346, `precipitation` 616, `shutdown` 346, `snow` 616, `user_activity_1` 346, `user_activity_2` 346, `holiday` 346, `school_holidays` 346, `shops_closed` 346, `winter_school_holidays` 346.
