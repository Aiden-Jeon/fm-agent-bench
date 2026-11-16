# Predict the potency of compounds against a protein target

## Description

Each row is one small-molecule compound that was tested against the same protein target. The features are a binary structural fingerprint of the molecule. The target, `MEDIAN_PXC50`, is the potency of the compound against the target: the negative log10 of the half-maximal concentration (pXC50, for example pIC50), taken as the median when the compound was measured several times. Higher values mean a more potent compound. Your task is to predict `MEDIAN_PXC50` for each compound in the test set.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the root mean squared error (RMSE) between the predicted and the observed `MEDIAN_PXC50`.

## Submission File

For each `id` in the test set, predict `MEDIAN_PXC50`. The file should contain a header and have the following format:

```
id,MEDIAN_PXC50
1,6.2
2,6.2
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 1,024 features and the target `MEDIAN_PXC50`
- `test.csv` — the test set: `id` and the same features. Predict `MEDIAN_PXC50` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `FCFP6_1024_0`, ..., `FCFP6_1024_1023` — a 1,024-bit folded circular fingerprint of the molecule (functional-class variant, diameter 6). Each column is 1 if one of the substructures hashed to that bit is present in the molecule, else 0
- `MEDIAN_PXC50` — target, mostly between 3.5 and 11

All features are 0/1 and there are no missing values.
