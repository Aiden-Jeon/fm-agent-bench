# Predict whether an access request is approved

## Description

Each row is a request by an employee of a large company for access to a resource (for example a software tool or a system). Your task is to predict, for each request in the test set, whether the access was approved, from the requested resource and the employee's role information.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `ResourceApproved` is `Yes`. The file should contain a header and have the following format:

```
id,ResourceApproved
0,0.5
3,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 9 features and the target `ResourceApproved`
- `test.csv` — the test set: `id` and the same features. Predict `ResourceApproved` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns (all features are anonymized integer codes: treat them as categories, the numbers have no order or magnitude meaning)

- `RESOURCE` — the requested resource
- `MGR_ID` — the employee id of the requester's manager
- `ROLE_ROLLUP_1` — role grouping category, first level
- `ROLE_ROLLUP_2` — role grouping category, second level
- `ROLE_DEPTNAME` — role department
- `ROLE_TITLE` — role business title
- `ROLE_FAMILY_DESC` — extended role family description
- `ROLE_FAMILY` — role family
- `ROLE_CODE` — role code (unique for each role)
- `ResourceApproved` — target: `Yes` if access was approved, `No` otherwise (about 6% `No`)

There are no missing values. Many category values are rare, and some values in the test set do not appear in the training set.
