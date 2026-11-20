# Predict which Android apps are malicious from the permissions they request

## Description

Each row is one Android app. The features record which permissions the app requests. Your task is to predict, for each app in the test set, whether the app is malware (`Yes`) or a benign app (`No`).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `Malware` is `Yes`. The file should contain a header and have the following format:

```
id,Malware
1,0.5
3,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 86 features and the target `Malware`
- `test.csv` — the test set: `id` and the same features. Predict `Malware` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Columns

- 86 permission columns. Each column name is a permission string as declared by the apps: standard Android permissions (for example `android.permission.INTERNET`, `android.permission.READ_SMS`, `android.permission.RECEIVE_BOOT_COMPLETED`) and permissions defined by device vendors, launchers, app stores or other apps (for example `com.huawei.android.launcher.permission.CHANGE_BADGE`, `com.android.vending.BILLING`, `com.google.android.c2dm.permission.RECEIVE`). The names are kept as they appear in the apps, including one malformed name (`android.permission.android.permission.READ_PHONE_STATE`)
- Each value is `1` if the app requests that permission and `0` otherwise
- `Malware` — target: `Yes` if the app is malware, `No` otherwise (about 35% `Yes`)

All features are binary and there are no missing values.
