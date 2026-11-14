# Contaminant detection from microwave sensor readings

## Description

Each row is one measurement of a closed food container taken with a microwave sensing system. Six antennas are placed around the container; for every ordered pair of different antennas, the system records how the signal sent by one antenna is received by the other, at a single frequency. Some containers hold a small foreign-body contaminant. Your task is to predict the class of each container in the test set (clean or contaminated) from its sensor readings.

Rows are independent measurements and are given in random order. The `id` column is a random identifier and carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `Contaminated` is `1`. The file should contain a header and have the following format:

```
id,Contaminated
0,0.5
3,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 30 sensor features and the target `Contaminated`
- `test.csv` — the test set: `id` and the same 30 features. Predict `Contaminated` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `s12`, `s13`, ..., `s65` — `sIJ` is the reading for the antenna pair (I, J), I ≠ J, with antennas numbered 1 to 6 (30 pairs). It is a scattering-parameter measurement (signal transmitted between antennas I and J). Values are small real numbers; their exact unit and preprocessing are not documented.
- `Contaminated` — binary class label (0 or 1) separating clean from contaminated containers. The two classes are about equally frequent.

All features are numeric and there are no missing values.
