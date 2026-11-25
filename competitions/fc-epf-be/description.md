# Forecast the hourly day-ahead electricity price

## Description

The single series is the hourly day-ahead price of electricity in one wholesale market, with two day-ahead forecasts published before the price is set. There is 1 series. You get the full hourly history of the series and must forecast the next 24 hours (the forecast horizon).

Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, …, 0.9 quantiles of the predictive distribution.

The `id` values are random identifiers that carry no information beyond grouping the rows of one series.

## Evaluation

Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and each series is scaled by its in-sample seasonal error: the mean absolute difference y_t − y_{t−24} over that series' history in `train.csv` (seasonal period 24). Series whose seasonal error is zero or undefined do not count.

The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a secondary metric.

## Submission File

One row per (`id`, `timestamp`) in `test.csv` (24 rows): the point forecast and the 9 quantiles. The file should contain a header and have the following format:

```
id,timestamp,predictions,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
s001,2016-12-31 00:00:00,50.78,50.78,50.78,50.78,50.78,50.78,50.78,50.78,50.78,50.78
s001,2016-12-31 01:00:00,50.78,50.78,50.78,50.78,50.78,50.78,50.78,50.78,50.78,50.78
...
```

## Dataset Description

- `train.csv` — the history: `id`, `timestamp`, `target` and 2 covariates. 52,392 rows, 52,392 hours per series, from 2011-01-09 to 2016-12-30
- `test.csv` — the rows to forecast: `id`, `timestamp` and the known covariates, 24 hours per series from 2016-12-31 to 2016-12-31
- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)

Columns

- `id` — series identifier
- `timestamp` — time of the observation (hourly, regular within each series)
- `target` — the value to forecast
- `Generation forecast` — known: day-ahead forecast of generation
- `System load forecast` — known: day-ahead forecast of the system load

"known" covariates are given for both the history and the forecast horizon (in `test.csv`). "past only" covariates are given in `train.csv` only; their future values are not available.

Missing values: 0 in `target`.
