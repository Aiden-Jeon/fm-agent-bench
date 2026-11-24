# Unsupervised anomaly detection on a numeric table

## Description

Each row is one observation described by 8 numeric features. A minority of the rows are anomalies, but no row is labeled.
`train.csv` is an unlabeled sample of 1038 rows; it contains anomalies at about the same (unknown) rate as the test set.
Your task is to give every row of `test.csv` an anomaly score: the higher the score, the more likely the row is an anomaly.

The features have no names and no documented meaning. Each feature was min-max scaled with the minimum and maximum of
`train.csv`, so test values can fall slightly outside [0, 1]. There are no missing values. Rows are independent and given
in random order; the `id` column is a random identifier and carries no information.

## Rules

- There are no labels anywhere: build an unsupervised detector.
- Fit everything (preprocessing, feature transforms, models, hyperparameters) on the rows of `train.csv` only. Do not fit,
  tune or transform anything using the rows of `test.csv`; score each test row with what you fitted on `train.csv`, as if
  the test rows arrived one at a time after training.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUROC) between the submitted score and the hidden anomaly label
of each test row. Only the ordering of the scores matters; any finite real number is a valid score.

## Submission File

For each `id` in the test set, give an anomaly score (higher = more anomalous). The file should contain a header and have
the following format:

```
id,score
1231,0.0
1083,0.0
...
```

## Dataset Description

- `train.csv` — the unlabeled training sample: `id` and the 8 features `f0`, ..., `f7` (1038 rows)
- `test.csv` — the rows to score: `id` and the same 8 features (446 rows)
- `sample_submission.csv` — a sample submission file in the correct format
