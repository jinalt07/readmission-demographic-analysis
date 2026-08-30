# Evaluating Demographic Differences in Machine Learning–Based Hospital Readmission Prediction

## Project Overview

Hospital readmission prediction models are commonly evaluated using
overall performance metrics, but aggregate results may hide differences
in how well a model performs across demographic groups.

This project evaluates whether a machine-learning model for 30-day
hospital readmission performs consistently across race, gender, and age
groups using the UCI Diabetes 130-US Hospitals dataset.

The analysis also investigates whether subgroup performance differences
coincide with differences in demographic representation and whether
race-based sample weighting changes model performance.

## Research Question

**Does a machine-learning model for 30-day hospital readmission perform
consistently across demographic groups, and how might differences in
group representation affect model performance?**

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008**
dataset from the UCI Machine Learning Repository.

The dataset contains hospital encounters for patients with diabetes and
includes:

- race
- gender
- age
- diagnoses
- medications
- hospitalization characteristics
- prior inpatient visits
- prior outpatient visits
- prior emergency visits
- readmission outcome

The original dataset contains 101,766 hospital encounters involving
71,518 unique patients.

## Prediction Target

The original `readmitted` variable was converted into a binary outcome:

- `<30` = 1
- `>30` or `NO` = 0

The model therefore predicts whether a hospital encounter will be
followed by readmission within 30 days.

## Methodology

### Data Preparation

- Replaced `"?"` values with missing values.
- Removed variables with substantial missingness that were not required
  for the analysis.
- Simplified diagnosis codes into broader clinical categories.
- Encoded categorical predictors using one-hot encoding.
- Standardized numerical predictors.
- Excluded `encounter_id` and `patient_nbr` from model predictors.

### Patient-Level Train/Test Split

Because individual patients may have multiple hospital encounters,
the data were split using patient ID rather than randomly by encounter.

This prevents encounters from the same patient from appearing in both
training and test datasets.

A GroupShuffleSplit was used with:

- 80% training
- 20% testing
- random state = 42

No patients overlapped between the training and test sets.

## Models

Two models were evaluated:

1. Logistic Regression
2. XGBoost

Class imbalance was addressed using balanced class weights for Logistic
Regression and `scale_pos_weight` for XGBoost.

Models were evaluated using:

- AUROC
- AUPRC
- Precision
- Recall
- F1 score
- Confusion matrix

## Demographic Subgroup Analysis

XGBoost performance was evaluated separately across:

### Race

- Caucasian
- African American
- Asian
- Hispanic
- Other
- Missing race information

### Gender

- Female
- Male

### Age

- 0–39
- 40–59
- 60–79
- 80–100

For each subgroup, the analysis reports representation, positive
readmission cases, AUROC, recall, precision, and F1 score.

## Representation Sensitivity Analysis

Race-based sample weighting was used as a sensitivity experiment to
investigate whether increasing the influence of less-represented racial
groups during training reduced subgroup performance differences.

The weighted model was compared with the original XGBoost model using
both overall and race-specific performance metrics.

## Model Interpretation

XGBoost feature importance was used to identify the predictors that
contributed most strongly to readmission prediction.

Important predictors included prior inpatient utilization,
hospitalization duration, number of diagnoses, age, medication status,
and diagnosis categories.

## Repository Structure

```text
readmission-demographic-analysis/
│
├── data/
│   └── README.md
│
├── figures/
│   ├── model_roc_compare.png
│   ├── model_precision_recall_compare.png
│   ├── race_representation.png
│   ├── race_representation_vs_recall.png
│   ├── race_weighting_recall_comparison.png
│   └── xgboost_feature_importance.png
│
├── results/
│   ├── model_compare_results.csv
│   ├── race_subgroup_results.csv
│   ├── gender_subgroup_results.csv
│   ├── age_subgroup_results.csv
│   ├── weighting_sensitivity_results.csv
│   └── xgboost_feature_importance.csv
│
├── src/
│   └── readmission_analysis.py
│
├── .gitignore
├── README.md
└── requirements.txt