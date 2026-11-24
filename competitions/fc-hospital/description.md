# Forecast monthly patient demand for medical products

## Description

Each series is the monthly count of patient demand for one medical product. There are 767 series. For every series you get its full monthly history up to the same forecast start and must forecast the next 12 months (the forecast horizon).

Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, …, 0.9 quantiles of the predictive distribution.

The `id` values are random identifiers that carry no information beyond grouping the rows of one series.

## Evaluation

Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and each series is scaled by its in-sample seasonal error: the mean absolute difference y_t − y_{t−12} over that series' history in `train.csv` (seasonal period 12). Series whose seasonal error is zero or undefined do not count.

The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a secondary metric.

## Submission File

One row per (`id`, `timestamp`) in `test.csv` (9,204 rows): the point forecast and the 9 quantiles. The file should contain a header and have the following format:

```
id,timestamp,predictions,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
s001,2006-01-31,745.0,745.0,745.0,745.0,745.0,745.0,745.0,745.0,745.0,745.0
s001,2006-02-28,745.0,745.0,745.0,745.0,745.0,745.0,745.0,745.0,745.0,745.0
...
```

## Dataset Description

- `train.csv` — the history: `id`, `timestamp`, `target`. 55,224 rows, 72 months per series, from 2000-01-31 to 2005-12-31
- `test.csv` — the rows to forecast: `id`, `timestamp`, 12 months per series from 2006-01-31 to 2006-12-31
- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)

Columns

- `id` — series identifier
- `timestamp` — time of the observation (monthly, regular within each series)
- `target` — the value to forecast

Missing values: 0 in `target`.
