# Predict which customers will buy a new product

## Description

Each row is one customer of a large company. The features are anonymized variables taken from the company's customer records. Your task is to predict, for each customer in the test set, whether the customer bought a new product or service offered by the company (`1`) or not (`-1`).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `BoughtNewProduct` is `1`. The file should contain a header and have the following format:

```
id,BoughtNewProduct
0,0.5
4,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 212 features and the target `BoughtNewProduct`
- `test.csv` — the test set: `id` and the same features. Predict `BoughtNewProduct` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns (the meaning of the variables is not given)

- `Var1`, ..., `Var190` — 174 anonymized numeric variables (the numbering has gaps)
- `Var191`, ..., `Var229` — 38 anonymized categorical variables; the values are scrambled strings with no order or meaning. Several of these columns have thousands of distinct values
- `BoughtNewProduct` — target: `1` if the customer bought the new product or service, `-1` otherwise (about 1.8% `1`)

Most numeric columns are mostly missing, and some categorical columns are mostly missing. Many category values are rare, and some values in the test set do not appear in the training set.
