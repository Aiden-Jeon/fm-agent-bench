# Forecast weekly influenza-like illness rates

## Description

Each series is the weekly rate of influenza-like illness (consultations per population, in its original units) reported by one region's surveillance system. There are 25 series. For every series you get its full weekly history and must forecast the next 13 weeks after the end of that history (the forecast horizon). Series end at different times, so the forecast horizons of different series are not all the same calendar period.

Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, …, 0.9 quantiles of the predictive distribution.

The `id` values are random identifiers that carry no information beyond grouping the rows of one series.

## Evaluation

Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and each series is scaled by its in-sample seasonal error: the mean absolute difference y_t − y_{t−1} over that series' history in `train.csv` (seasonal period 1). Series whose seasonal error is zero or undefined do not count.

The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a secondary metric.

## Submission File

One row per (`id`, `timestamp`) in `test.csv` (325 rows): the point forecast and the 9 quantiles. The file should contain a header and have the following format:

```
id,timestamp,predictions,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
s001,2025-05-18,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2
s001,2025-05-25,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2
...
```

## Dataset Description

- `train.csv` — the history: `id`, `timestamp`, `target`. 4,472 rows, 76 to 203 (median 188) weeks per series, from 2021-06-27 to 2025-05-11
- `test.csv` — the rows to forecast: `id`, `timestamp`, 13 weeks per series from 2022-12-11 to 2025-08-10
- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)

Columns

- `id` — series identifier
- `timestamp` — time of the observation (weekly, regular within each series)
- `target` — the value to forecast

Missing values: 0 in `target`.
