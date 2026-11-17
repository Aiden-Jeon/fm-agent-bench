# Predict truck component failures from operational data

## Description

Each row is one failure event of a heavy truck in daily use. The features are readings from the truck's operational data, recorded up to the failure. Your task is to predict, for each event in the test set, whether the failed component belongs to the air pressure system (`pos`) or to another system of the truck (`neg`). The air pressure system produces pressurized air used for functions such as braking and gear changes.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `AirPressureSystemFailure` is `pos`. The file should contain a header and have the following format:

```
id,AirPressureSystemFailure
0,0.5
2,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 170 features and the target `AirPressureSystemFailure`
- `test.csv` — the test set: `id` and the same features. Predict `AirPressureSystemFailure` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `aa_000`, ..., `eg_000` — 170 anonymized operational counters (the meaning of each sensor is not given). Most columns are single numeric counters. Seven groups are histograms with ten bins each: `ag_000`–`ag_009`, `ay_000`–`ay_009`, `az_000`–`az_009`, `ba_000`–`ba_009`, `cn_000`–`cn_009`, `cs_000`–`cs_009` and `ee_000`–`ee_009`; within a group, the columns are the counts in consecutive bins of one measured quantity
- `AirPressureSystemFailure` — target: `pos` for an air pressure system failure, `neg` otherwise (about 1.8% `pos`)

All features are numeric. Almost all columns have missing values, some columns are mostly missing.
