# Multi-class classification on anonymized categorical data

## Description

This is a multi-class classification task on an anonymized table. The column names and the category values have been replaced by codes, and no information about what the rows or columns represent is available. Your task is to predict, for each row in the test set, the class of `target`: `c0`, `c1` or `c2`.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the multi-class logarithmic loss between the predicted class probabilities and the observed class. Probabilities are clipped to [1e-15, 1] and each row is rescaled to sum to 1 before scoring.

## Submission File

For each `id` in the test set, predict a probability for each of the three classes. The file should contain a header and have the following format:

```
id,c0,c1,c2
1,0.333,0.333,0.334
3,0.333,0.333,0.334
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 60 features and the target `target`
- `test.csv` — the test set: `id` and the same features. Predict `target` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `f0`, `f1`, ..., `f59` — anonymized features. All 60 are categorical. In each column, every category value has been replaced by a random 5-character code. Codes are assigned independently in each column: the same code in two different columns does not mean the same thing
- `target` — the class to predict: `c0`, `c1` or `c2`
