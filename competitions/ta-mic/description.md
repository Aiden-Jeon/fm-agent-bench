# Predict the outcome of patients hospitalized with a heart attack

## Description

Each row is one patient admitted to hospital with an acute myocardial infarction (heart attack). The features describe the patient's medical history, the state at admission (blood pressure, ECG findings, laboratory values) and the treatment and relapses of pain during the first days in hospital. Your task is to predict, for each patient in the test set, whether the patient survived (`alive`) or, if not, the cause of death (seven classes).

Rows are given in random order and the `id` column is a random identifier that carries no information.

## Evaluation

Submissions are evaluated on the multi-class logarithmic loss between the predicted class probabilities and the observed class. Probabilities are clipped to [1e-15, 1] and each row is rescaled to sum to 1 before scoring.

## Submission File

For each `id` in the test set, predict a probability for each of the eight classes. The file should contain a header and have the following format:

```
id,alive,asystole,cardiogenic_shock,myocardial_rupture,progress_congestive_heart_failure,pulmonary_edema,thromboembolism,ventricular_fibrillation
0,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125
2,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125
...
```

## Dataset Description

- `train.csv` — the training set: `id`, 111 features and the target `LET_IS`
- `test.csv` — the test set: `id` and the same features. Predict `LET_IS` for these rows
- `sample_submission.csv` — a sample submission file in the correct format

The column names are abbreviations. Unless stated otherwise, a column is a 0/1 flag (1 = present / given).

Patient and history

- `AGE` (years), `SEX` (0 = female, 1 = male)
- `INF_ANAM` — number of earlier heart attacks (0, 1, 2, 3 or more)
- `STENOK_AN` — exertional angina in the history (0 = never, 1 = during the last year, 2 = one year ago, ..., 6 = more than 5 years ago)
- `FK_STENOK` — functional class of angina during the last year (0 = none, 1–4 = class I–IV)
- `IBS_POST` — coronary heart disease in the recent weeks before admission (0 = none, 1 = exertional angina, 2 = unstable angina)
- `IBS_NASL` — family history of coronary heart disease
- `GB` — essential hypertension (0 = none, 1–3 = stage); `SIM_GIPERT` — symptomatic hypertension; `DLIT_AG` — how long the patient has had high blood pressure (0 = none, 1–7 = increasing duration)
- `ZSN_A` — chronic heart failure in the history (0 = none, 1–4 = stage)
- `nr_11` — arrhythmia in the history; `nr_01`, `nr_02`, `nr_03`, `nr_04`, `nr_07`, `nr_08` — specific rhythm disorders in the history (premature atrial / ventricular contractions, paroxysmal / persistent atrial fibrillation, ventricular fibrillation, paroxysmal ventricular tachycardia)
- `np_01`, `np_04`, `np_05`, `np_07`, `np_08`, `np_09`, `np_10` — specific conduction disorders (heart blocks, bundle branch blocks) in the history
- `endocr_01` (diabetes), `endocr_02` (obesity), `endocr_03` (thyrotoxicosis)
- `zab_leg_01`, `zab_leg_02`, `zab_leg_03`, `zab_leg_04`, `zab_leg_06` — chronic lung diseases (chronic bronchitis, obstructive chronic bronchitis, asthma, chronic pneumonia, tuberculosis)

At admission

- `S_AD_KBRIG`, `D_AD_KBRIG` — systolic and diastolic blood pressure (mmHg) measured by the emergency cardiology team; `S_AD_ORIT`, `D_AD_ORIT` — the same measured in the intensive care unit
- `O_L_POST` (pulmonary edema), `K_SH_POST` (cardiogenic shock), `MP_TP_POST` (paroxysm of atrial fibrillation), `SVT_POST` (paroxysm of supraventricular tachycardia), `GT_POST` (paroxysm of ventricular tachycardia), `FIB_G_POST` (ventricular fibrillation) — at the time of admission to the intensive care unit
- `ant_im`, `lat_im`, `inf_im`, `post_im` — presence and extent of infarction in the anterior, lateral, inferior and posterior wall of the left ventricle on the ECG (0 = none, 1–4 = increasing extent); `IM_PG_P` — right ventricular infarction
- `ritm_ecg_p_01`, `_02`, `_04`, `_06`, `_07`, `_08` — heart rhythm on the admission ECG (sinus 60–90/min, atrial fibrillation, atrial, idioventricular, sinus above 90/min, sinus below 60/min)
- `n_r_ecg_p_01`, ..., `n_r_ecg_p_10` — rhythm disorders on the admission ECG; `n_p_ecg_p_01`, ..., `n_p_ecg_p_12` — conduction disorders on the admission ECG
- `fibr_ter_01`, ..., `fibr_ter_08` — fibrinolytic (clot-dissolving) therapy with specific drugs
- `GIPO_K` — low serum potassium (below 4 mmol/L); `K_BLOOD` — serum potassium (mmol/L); `GIPER_NA` — high serum sodium (above 150 mmol/L); `NA_BLOOD` — serum sodium (mmol/L)
- `ALT_BLOOD`, `AST_BLOOD` — serum liver enzymes ALT and AST; `KFK_BLOOD` — serum creatine kinase (almost always missing); `L_BLOOD` — white blood cell count (10⁹/L); `ROE` — erythrocyte sedimentation rate (mm/h)
- `TIME_B_S` — time from the start of the attack to arrival at hospital (1 = less than 2 hours, 2 = 2–4 hours, ..., 9 = more than 3 days)

First days in hospital

- `R_AB_1_n`, `R_AB_2_n`, `R_AB_3_n` — relapses of pain on the 1st, 2nd and 3rd day (0 = none, 1, 2, 3 or more)
- `NA_KB`, `NOT_NA_KB`, `LID_KB` — opioid drugs, non-steroidal anti-inflammatory drugs and lidocaine given by the emergency team
- `NITR_S` — liquid nitrates in the intensive care unit
- `NA_R_1_n`, `NA_R_2_n`, `NA_R_3_n` — opioid drugs in the intensive care unit on the 1st, 2nd and 3rd day (number of times); `NOT_NA_1_n`, `NOT_NA_2_n`, `NOT_NA_3_n` — the same for non-steroidal anti-inflammatory drugs
- `LID_S_n` (lidocaine), `B_BLOK_S_n` (beta blockers), `ANT_CA_S_n` (calcium channel blockers), `GEPAR_S_n` (anticoagulants), `ASP_S_n` (aspirin), `TIKL_S_n` (ticlopidine), `TRENT_S_n` (pentoxifylline) — given in the intensive care unit

Target

- `LET_IS` — `alive` (about 84%) or the cause of death: `cardiogenic_shock`, `myocardial_rupture`, `asystole`, `ventricular_fibrillation`, `progress_congestive_heart_failure`, `pulmonary_edema`, `thromboembolism` (each 0.7–6%)

Most features are categorical codes (flags and ordered levels); `AGE`, the blood pressure and laboratory values and the `NA_R_*`, `NOT_NA_2_n`, `NOT_NA_3_n` counts are numeric. Every column except `SEX` has missing values (from a few rows to almost all rows).
