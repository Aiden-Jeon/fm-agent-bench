# Forecast the hourly workload of database clusters

## Description

Each series is the hourly workload (an aggregate activity measure, in its original units) of one database cluster of a cloud data warehouse service. There are 138 series. For every series you get its full hourly history and must forecast the next 24 hours after the end of that history (the forecast horizon). Series end at different times, so the forecast horizons of different series are not all the same calendar period.

Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, …, 0.9 quantiles of the predictive distribution.

The `id` values are random identifiers that carry no information beyond grouping the rows of one series.

## Evaluation

Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and each series is scaled by its in-sample seasonal error: the mean absolute difference y_t − y_{t−24} over that series' history in `train.csv` (seasonal period 24). Series whose seasonal error is zero or undefined do not count.

The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a secondary metric.

## Submission File

One row per (`id`, `timestamp`) in `test.csv` (3,312 rows): the point forecast and the 9 quantiles. The file should contain a header and have the following format:

```
id,timestamp,predictions,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
s001,2024-05-29 00:00:00,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0
s001,2024-05-29 01:00:00,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0,1581.0
...
```

## Dataset Description

- `train.csv` — the history: `id`, `timestamp`, `target` and 1 covariate. 279,758 rows, 336 to 2,162 (median 2,136) hours per series, from 2024-02-29 to 2024-05-30
- `test.csv` — the rows to forecast: `id`, `timestamp`, 24 hours per series from 2024-04-04 to 2024-05-31
- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)

Columns

- `id` — series identifier
- `timestamp` — time of the observation (hourly, regular within each series)
- `target` — the value to forecast
- `subset` — static: how the cluster is deployed (`provisioned` or `serverless`), the same value on every row of a series

Missing values: 0 in `target`.
