# Classify exon/intron boundaries in DNA sequences

## Description

Each row is a window of 60 nucleotides taken from a DNA sequence of a gene, centred on a candidate position. In a gene, the coding parts (exons) are separated by non-coding parts (introns). Your task is to predict, for each window in the test set, whether its centre is an exon→intron boundary (`EI`), an intron→exon boundary (`IE`), or neither (`N`).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the multi-class logarithmic loss between the predicted class probabilities and the observed class. Probabilities are clipped to [1e-15, 1] and each row is rescaled to sum to 1 before scoring.

## Submission File

For each `id` in the test set, predict a probability for each of the three classes. The file should contain a header and have the following format:

```
id,EI,IE,N
2,0.333,0.333,0.334
4,0.333,0.333,0.334
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 60 features and the target `SiteType`
- `test.csv` — the test set: `id` and the same features. Predict `SiteType` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `position_-30`, ..., `position_-1`, `position_1`, ..., `position_30` — the nucleotide at each position of the window, relative to the candidate boundary (there is no position 0: the boundary lies between `position_-1` and `position_1`). Values are `A`, `C`, `G`, `T`; a few positions hold ambiguity codes (`N` = any nucleotide, `D` = A, G or T, `R` = A or G, `S` = C or G)
- `SiteType` — target: `EI` (about 24%), `IE` (about 24%) or `N` (about 52%)

All features are categorical and there are no missing values.
