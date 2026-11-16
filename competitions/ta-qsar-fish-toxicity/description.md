# Predict the acute aquatic toxicity of chemicals

## Description

Each row is one chemical compound. The features are six molecular descriptors computed from the structure of the molecule. The target, `LC50`, is the measured acute toxicity of the compound towards a small freshwater fish species: the concentration that kills half of the test animals within 96 hours, given as −log(mol/L). Higher values mean a more toxic compound. Your task is to predict `LC50` for each compound in the test set.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the root mean squared error (RMSE) between the predicted and the observed `LC50`.

## Submission File

For each `id` in the test set, predict `LC50`. The file should contain a header and have the following format:

```
id,LC50
7,4.0
8,4.0
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 6 features and the target `LC50`
- `test.csv` — the test set: `id` and the same features. Predict `LC50` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `CIC0` — information index (complementary information content of order 0), a topological descriptor
- `SM1_Dz(Z)` — a spectral moment of the 2D distance matrix weighted by atomic number
- `GATS1i` — Geary autocorrelation of lag 1 weighted by ionization potential
- `NdsCH` — number of carbon atoms of the type `=CH-` (integer count)
- `NdssC` — number of carbon atoms of the type `=C<` (integer count)
- `MLOGP` — estimated octanol–water partition coefficient (lipophilicity, Moriguchi method)
- `LC50` — target, roughly between 0 and 10

All features are numeric and there are no missing values.
