# Predict early hospital readmission of patients with diabetes

## Description

Each row is one hospital stay (inpatient encounter) of a patient with diabetes, taken from clinical records of hospitals in the United States. Each patient appears at most once. Your task is to predict, for each stay in the test set, whether the patient was readmitted to a hospital within 30 days of discharge (`Yes`) or not (`No`), from the patient's demographics, the admission, the diagnoses, the procedures and laboratory tests during the stay, and the diabetes medications.

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the area under the ROC curve (AUC) between the predicted probability and the observed target.

## Submission File

For each `id` in the test set, predict the probability that `EarlyReadmission` is `Yes`. The file should contain a header and have the following format:

```
id,EarlyReadmission
1,0.5
5,0.5
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 47 features and the target `EarlyReadmission`
- `test.csv` — the test set: `id` and the same features. Predict `EarlyReadmission` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Patient columns

- `race`, `gender`, `age` (10-year bands such as `[70-80)`), `weight` (25-pound bands, mostly missing)
- `payer_code` — the patient's insurance or payer (short codes, for example `MC`)

Admission columns

- `admission_type_id` — for example `Emergency`, `Urgent`, `Elective`
- `admission_source_id` — where the patient came from, for example `Emergency Room`, `Physician Referral`
- `discharge_disposition_id` — where the patient went at discharge, for example `Discharged to home`, `Discharged/transferred to SNF` (skilled nursing facility), `Expired`
- `medical_specialty` — specialty of the admitting physician (about 70 values)
- `time_in_hospital` — length of the stay in days (1–14)

Counts for the stay and for the year before it

- `num_lab_procedures`, `num_procedures` (other than lab tests), `num_medications` — during the stay
- `number_outpatient`, `number_emergency`, `number_inpatient` — visits of each kind in the year before the stay
- `number_diagnoses` — number of diagnoses entered for the stay

Diagnoses

- `diag_1` (primary), `diag_2` (secondary), `diag_3` (additional secondary) — ICD-9 diagnosis codes, for example `428`, `250.02`, `V45`, `E885`. Each column has about 700 distinct codes, many of them rare, and some codes in the test set do not appear in the training set

Tests and medications

- `max_glu_serum` — glucose serum test result (`Norm`, `>200`, `>300`), `A1Cresult` — HbA1c test result (`Norm`, `>7`, `>8`). `nan` means the test was not done
- 23 medication columns (`metformin`, `repaglinide`, ..., `insulin`, `glyburide-metformin`, ..., `metformin-pioglitazone`) — `No` if the drug was not prescribed, otherwise whether the dose went `Up`, went `Down` or stayed `Steady` during the stay. Some drugs are almost never prescribed
- `change` — `Ch` if any diabetes medication was changed, `No` otherwise; `diabetesMed` — `Yes` if any diabetes medication was prescribed
- `EarlyReadmission` — target: `Yes` if the patient was readmitted within 30 days, `No` otherwise (about 9% `Yes`)

The counts and `time_in_hospital` are numeric; all other features are categorical (text values). Missing values appear in two ways: as empty cells (for example in `race`, `weight`, `payer_code`, `medical_specialty` and the diagnosis columns) and as the text `nan` (in the admission columns and the two test columns). The admission columns also use codes such as `Not Available` and `Not Mapped`.
