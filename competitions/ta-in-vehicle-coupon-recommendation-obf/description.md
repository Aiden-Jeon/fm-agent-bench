# Binary classification on anonymized mixed data

## Description

This is a binary classification task on an anonymized table. The column names and the category values have been replaced by codes, and no information about what the rows or columns represent is available. Your task is to predict, for each row in the test set, the probability that `target` is `c1` (the other class is `c0`).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `target` is `c1`. The file should contain a header and have the following format:

```
id,target
2,0.5
3,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 24 features and the target `target`
- `test.csv` — the test set: `id` and the same features. Predict `target` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `f0`, `f1`, ..., `f23` — anonymized features. 22 are categorical: all except `f3` and `f4`, which are numeric (their values are unchanged). In each categorical column, every category value has been replaced by a random 5-character code. Codes are assigned independently in each column: the same code in two different columns does not mean the same thing. Some categorical values are missing (empty cells)
- `target` — `c1` or `c0`
