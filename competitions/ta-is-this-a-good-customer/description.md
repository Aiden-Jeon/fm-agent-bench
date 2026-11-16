# Predict which borrowers become bad clients

## Description

Each row is one consumer loan that a bank gave to a customer to buy a product in a shop (for example a phone or a piece of furniture). The features describe the loan and the customer at the time of the application. Your task is to predict, for each loan in the test set, whether the customer turned out to be a bad client (did not repay the loan as agreed).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `bad_client_target` is `Yes`. The file should contain a header and have the following format:

```
id,bad_client_target
2,0.5
3,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 13 features and the target `bad_client_target`
- `test.csv` — the test set: `id` and the same features. Predict `bad_client_target` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- `month` — month of the year when the loan was issued (1–12)
- `credit_amount` — loan amount
- `credit_term` — loan term in months
- `age` — age of the customer in years
- `sex` — `male` or `female`
- `education` — highest education level of the customer
- `product_type` — kind of product bought with the loan (for example `Cell phones`, `Furniture`, `Tourism`)
- `having_children_flg` — 1 if the customer has children, else 0
- `region` — region of the customer (anonymized code 0, 1 or 2)
- `income` — income of the customer
- `family_status` — `Married`, `Unmarried` or `Another`
- `phone_operator` — mobile phone operator of the customer (anonymized code 0–4)
- `is_client` — 1 if the customer already was a client of the bank before this loan, else 0
- `bad_client_target` — target: `Yes` for a bad client, `No` otherwise (about 11% `Yes`)

`sex`, `education`, `product_type`, `having_children_flg`, `region`, `family_status`, `phone_operator` and `is_client` are categorical; the other features are numeric. There are no missing values.
