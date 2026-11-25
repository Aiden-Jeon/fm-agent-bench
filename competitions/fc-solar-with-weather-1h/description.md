# Forecast the hourly output of a solar power installation from weather

## Description

The single series is the hourly energy output (in its original units) of one solar power installation, together with hourly weather variables. There is 1 series. You get the full hourly history of the series and must forecast the next 24 hours (the forecast horizon).

Your forecast is probabilistic: for every series and future time step, give a point forecast and the 0.1, 0.2, …, 0.9 quantiles of the predictive distribution.

The `id` values are random identifiers that carry no information beyond grouping the rows of one series.

## Evaluation

Submissions are evaluated on the **scaled quantile loss (SQL)** averaged over the 9 quantile levels, all series and all horizon steps (lower is better). For quantile level τ the loss is 2·|(y − q̂_τ)·(1[y ≤ q̂_τ] − τ)|, and each series is scaled by its in-sample seasonal error: the mean absolute difference y_t − y_{t−24} over that series' history in `train.csv` (seasonal period 24). Series whose seasonal error is zero or undefined do not count.

The point forecast (`predictions`) is also scored with the mean absolute scaled error (MASE, same scaling) as a secondary metric.

## Submission File

One row per (`id`, `timestamp`) in `test.csv` (24 rows): the point forecast and the 9 quantiles. The file should contain a header and have the following format:

```
id,timestamp,predictions,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
s001,2022-08-30 17:00:00,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0
s001,2022-08-30 18:00:00,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0,1150.0
...
```

## Dataset Description

- `train.csv` — the history: `id`, `timestamp`, `target` and 9 covariates. 49,624 rows, 49,624 hours per series, from 2017-01-01 to 2022-08-30
- `test.csv` — the rows to forecast: `id`, `timestamp` and the known covariates, 24 hours per series from 2022-08-30 to 2022-08-31
- `sample_submission.csv` — a sample submission file in the correct format (it repeats each series' last value)

Columns

- `id` — series identifier
- `timestamp` — time of the observation (hourly, regular within each series)
- `target` — the value to forecast
- `day_length` — known: length of the day (minutes)
- `humidity` — known: relative humidity (%)
- `pressure` — known: air pressure (hPa)
- `rain_1h` — known: rain in the hour
- `snow_1h` — known: snow in the hour
- `temp` — known: air temperature (°C)
- `wind_speed` — known: wind speed
- `clouds_all` — past only: cloud cover (%)
- `global_horizontal_irradiance` — past only: global horizontal solar irradiance

"known" covariates are given for both the history and the forecast horizon (in `test.csv`). "past only" covariates are given in `train.csv` only; their future values are not available.

Missing values: 0 in `target`; covariates: `clouds_all` 456, `global_horizontal_irradiance` 456, `day_length` 456, `humidity` 456, `pressure` 456, `rain_1h` 456, `snow_1h` 456, `temp` 456, `wind_speed` 456.
