# Predict whether a university student drops out, is still enrolled, or graduates

## Description

Each row is one student enrolled in an undergraduate degree at a higher-education institution. The features are what was known at enrollment (demographics, socio-economic background, how the student was admitted) and the student's academic results in the first and second semesters. Your task is to predict, for each student in the test set, the outcome at the end of the normal duration of the degree: dropped out (`Dropout`), still enrolled (`Enrolled`) or graduated (`Graduate`).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the multi-class logarithmic loss between the predicted class probabilities and the observed class. Probabilities are clipped to [1e-15, 1] and each row is rescaled to sum to 1 before scoring.

## Submission File

For each `id` in the test set, predict a probability for each of the three classes. The file should contain a header and have the following format:

```
id,Dropout,Enrolled,Graduate
2,0.333,0.333,0.334
5,0.333,0.333,0.334
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 36 features and the target `AcademicOutcome`
- `test.csv` — the test set: `id` and the same features. Predict `AcademicOutcome` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

Student and background columns

- `Marital_status`, `Nationality`, `Gender`, `Age_at_enrollment`
- `Displaced` — the student lives away from home; `Educational_special_needs`; `International` — international student
- `Mothers_qualification`, `Fathers_qualification` — highest education level of the parents; `Mothers_occupation`, `Fathers_occupation`
- `Debtor` — the student owes money to the institution; `Tuition_fees_up_to_date`; `Scholarship_holder`

Admission columns

- `Application_mode` — admission route (for example general contingent, over 23 years old, change of course)
- `Application_order` — rank of this degree in the student's list of choices (0 = first choice)
- `Course` — the degree programme
- `Daytimeevening_attendance` — `daytime` or `evening` classes
- `Previous_qualification`, `Previous_qualification_grade` — qualification before enrollment and its grade (0–200)
- `Admission_grade` — admission grade (0–200)

Academic results (for the 1st and the 2nd semester, `Curricular_units_1st_sem_*` and `Curricular_units_2nd_sem_*`)

- `_credited` — curricular units credited from earlier studies
- `_enrolled` — units the student enrolled in
- `_evaluations` — number of evaluations (exams) taken
- `_approved` — units passed
- `_grade` — average grade (0–20)
- `_without_evaluations` — units enrolled in but without any evaluation

Economic context at the time of enrollment

- `Unemployment_rate` (%), `Inflation_rate` (%), `GDP` (growth, %)

Target

- `AcademicOutcome` — `Dropout` (about 32%), `Enrolled` (about 18%) or `Graduate` (about 50%)

`Marital_status`, `Application_mode`, `Course`, `Daytimeevening_attendance`, `Previous_qualification`, `Nationality`, the parents' qualification and occupation columns, and the yes/no columns are categorical; the other features are numeric. There are no missing values.
