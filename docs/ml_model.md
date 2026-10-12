# ML Model: 30-Day Readmission Risk

> Personal portfolio project. Built to show an honest ML workflow on top of the Gold layer. **Not validated for any clinical use.**

## Purpose

Estimate, at the moment of discharge, how likely a patient is to be readmitted within 30 days, so patients can be ranked into relative risk bands. The model ranks patients. It does not output reliable probabilities (see Limitations).

## Data

| Item | Value |
|------|-------|
| Source | `gold/fact_encounter` (Delta) |
| Population | 99,320 encounters (69,979 distinct patients, 1.42 encounters per patient) |
| Excluded | 2,422 encounters flagged `not_at_risk_of_readmission` (discharged as expired or to hospice), because they cannot realistically be readmitted |
| Target | `readmitted_30d` (1 = readmitted within 30 days, 0 = otherwise) |
| Positive class | 11,309 encounters (11.4%) |

Note: 43 of the 2,422 excluded encounters are marked readmitted in the source data. This is a quirk of the source and was left as is.

## Features

The prediction moment is discharge, so every feature must be known by then.

- **Numeric (9):** `age_midpoint` (midpoint of the age band, e.g. 75 for `[70-80)`), `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`
- **Categorical (9):** `gender`, `race`, `max_glu_serum`, `A1Cresult`, `change`, `diabetesMed`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`. The three ID columns are treated as categories, not numbers.

**Deliberately excluded:**

| Column | Reason |
|--------|--------|
| `readmitted` | Text version of the target (direct leakage) |
| `readmitted_30d` | The target itself |
| `encounter_id` | Identifier, no meaning for the model |
| `patient_nbr` | Used only to group the train/test split, never as a feature |

## Method

1. **Split by patient, not by row.** 80/20 using grouped shuffling on `patient_nbr` (seed 42). Train 79,596 rows, test 19,724 rows. Verified that 0 patients appear in both sets, and that the readmission rate is similar in both (11.4% and 11.3%).
2. **Preprocessing inside a pipeline.** Numeric features standardised, categorical features one-hot encoded (categories with fewer than 50 training rows grouped into an "infrequent" bucket). Fitting inside the pipeline means test data never influences preprocessing.
3. **Class imbalance.** Handled with balanced class weights. Accuracy was not used as a metric, because always predicting "not readmitted" would score about 89%.
4. **Metrics.** ROC-AUC (ranking quality, 0.5 = guessing) and PR-AUC (more informative for rare outcomes; guessing scores about 0.113).

## Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic regression (chosen) | 0.653 | 0.202 |
| Gradient boosting | 0.658 | 0.209 |
| Random guessing | 0.500 | 0.113 |

Gradient boosting improved the scores by about 0.005 to 0.007, which is similar in size to the random noise in these metrics on a test set this size. The much more flexible model barely helped, which suggests the limit is in the data (hospital records omit things like home circumstances and follow-up care), not in the algorithm. **Logistic regression was kept** because it is simpler and its coefficients can be inspected.

**Practical value (test set, top 10% of scores):** readmission rate of 25.0% against 11.3% overall, capturing 22.2% of all readmissions. That is a lift of about 2.2 times over random selection. About three in four patients in the highest-scored group are still not readmitted.

## Scoring every patient

Risk scores for all 99,320 encounters were produced with 5-fold cross-validation grouped by patient, so each patient is scored by a model that never saw them. Out-of-fold ROC-AUC: 0.660.

Bands are assigned by rank: top 10% = High, next 20% = Medium, remaining 70% = Low.

| Band | Encounters | Actual 30-day readmission rate |
|------|-----------|-------------------------------|
| High | 9,932 | 25.9% |
| Medium | 19,864 | 15.3% |
| Low | 69,524 | 8.2% |

Output table: `gold/readmission_risk` (Delta) with `encounter_id`, `risk_score`, `risk_band`. It is shown on the `Model Risk` page of the Power BI dashboard. Patients excluded from modelling appear as "(Blank)" and are filtered out of that chart.

## What drives the model

Strongest influences (logistic regression coefficients):

- **Raise risk:** transfer to a psychiatric hospital, transfer to a rehab facility, number of prior inpatient visits
- **Lower risk:** transfer to a long-term care hospital, discharge home

Prior inpatient visits show a clear pattern in the raw data too (the readmission rate rises from under 10% with no prior visits to over 40% at the highest visit counts).

## Limitations

- **Weak signal.** ROC-AUC of about 0.66 is modest. The model is useful for ranking, not for individual decisions.
- **Scores are not probabilities.** Balanced class weights inflate raw scores, so a score of 0.6 does not mean a 60% chance. Bands are based on rank for this reason. No calibration step was applied.
- **Recording artefacts.** Placeholder codes ("Not Mapped" discharge, "Not Available" and NULL admission sources) carry weight in the model, so it partly learns how records were completed, not only clinical risk.
- **Discharge destination dominates.** Several of the largest coefficients are discharge codes. These are associations, not causes: a transfer destination lowering the score does not mean the transfer prevents readmission.
- **Old, US-only data (1999 to 2008, 130 hospitals).** May not reflect current practice or other health systems.
- **No tuning.** Hyperparameters were not searched. Given how little the more flexible model helped, large gains from tuning are unlikely.
- **Prior-visit features** are described in the dataset documentation as counts from the preceding year. This was not independently verified.

## Responsible AI considerations

- **Sensitive inputs.** `race` and `gender` are included as model inputs. No fairness audit was performed (performance and risk-band assignment were not compared across groups). Before any real use, this would need checking, and removing race as an input would be worth testing.
- **Cost of errors.** In healthcare a missed readmission (false negative) is usually costlier than an extra follow-up call (false positive). The bands support ranking by risk rather than a single yes/no cut-off, so the threshold can be set to suit the cost of each error.
- **Excluded group.** Patients who died or went to hospice are not scored.
- **Human oversight.** The output is decision support for ranking, not an automated decision.
- **Not clinically validated.** This is a portfolio exercise on public, historical data.

## Reproducibility

Notebook: `notebooks/04_ml_readmission`. Random seed 42 for the split. Requires the Gold Delta tables from the earlier notebooks.
