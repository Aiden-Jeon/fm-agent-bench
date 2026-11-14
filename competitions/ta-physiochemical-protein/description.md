# Predict how far a protein structure model is from the true structure

## Description

Each row is one computer-generated 3D model of a protein structure. The features are physicochemical properties computed from the model. The target, `ResidualSize`, measures how far the model is from the experimentally determined true structure (a root-mean-square deviation, in ångström). Your task is to predict `ResidualSize` for each model in the test set.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the root mean squared error (RMSE) between the predicted and the observed `ResidualSize`.

## Submission File

For each `id` in the test set, predict `ResidualSize`. The file should contain a header and have the following format:

```
id,ResidualSize
1,7.7
18,7.7
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 9 features and the target `ResidualSize`
- `test.csv` — the test set: `id` and the same features. Predict `ResidualSize` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `TotalSurfaceArea` — total surface area
- `NonPolarExposedArea` — non-polar exposed area
- `FracExposedNonPolarResidue` — fractional area of exposed non-polar residues
- `FracExposedNonPolarPart` — fractional area of the exposed non-polar part of residues
- `MassWeightedExposedArea` — molecular-mass-weighted exposed area
- `AvgDeviationExposedArea` — average deviation from the standard exposed area of residues
- `EuclideanDistance` — Euclidean distance
- `SecondaryStructurePenalty` — secondary structure penalty (integer)
- `SpatialDistNK` — spatial distribution constraint (N, K value)
- `ResidualSize` — target, between 0 and 21

All features are numeric and there are no missing values.
