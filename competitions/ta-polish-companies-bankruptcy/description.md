# Predict company bankruptcy from financial ratios

## Description

Each row is one company. The features are financial ratios computed from the company's financial statements for one year. Your task is to predict, for each company in the test set, whether it went bankrupt within the following year.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `company_bankrupt` is `Yes`. The file should contain a header and have the following format:

```
id,company_bankrupt
0,0.5
2,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 64 features and the target `company_bankrupt`
- `test.csv` — the test set: `id` and the same features. Predict `company_bankrupt` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- 64 financial ratios. The column names say what each ratio is, for example `net_profit_to_total_assets` (net profit / total assets), `working_capital_to_total_assets`, `current_assets_to_short_term_liabilities`, `log_total_assets` (logarithm of total assets), `sales_growth_ratio` (sales this year / sales the year before). Columns ending in `_days_ratio` are ratios multiplied by 365 (for example `receivables_days_ratio` = receivables × 365 / sales). `working_capital_absolute` is working capital in money units, not a ratio
- `company_bankrupt` — target: `Yes` if the company went bankrupt, `No` otherwise (about 7% `Yes`)

All features are numeric. Many columns have missing values (a ratio could not be computed, for example because a denominator was missing). Some ratios have extreme outliers.
