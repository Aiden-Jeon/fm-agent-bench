# Predict which molecules are active from their numeric descriptors

## Description

Each row is one molecule. The features are numeric descriptors computed from the molecule's structure. Your task is to predict, for each molecule in the test set, whether the molecule was active in a laboratory experiment (`Yes`) or not (`No`).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `Active` is `Yes`. The file should contain a header and have the following format:

```
id,Active
1,0.5
7,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 1,776 features and the target `Active`
- `test.csv` — the test set: `id` and the same features. Predict `Active` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns (the meaning of the individual descriptors is not given)

- `D1`, ..., `D1776` — 1,776 numeric molecular descriptors, all scaled to the range 0 to 1. 834 of them take only the values `0` and `1`; the others are continuous or take a small number of distinct values
- `Active` — target: `Yes` if the molecule was active, `No` otherwise (about 54% `Yes`)

There are no missing values.
