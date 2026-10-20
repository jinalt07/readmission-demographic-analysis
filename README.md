### Evaluating Demographic Differences in Machine Learning–Based Hospital Readmission Prediction

### Project Overview

Hospital readmission prediction models are commonly evaluated using overall performance metrics, but aggregate results may hide differences in how well a model performs across demographic groups.

This project evaluates whether a machine-learning model for 30-day hospital readmission performs consistently across race, gender, and age groups using the UCI Diabetes 130-US Hospitals dataset.

The analysis also investigates whether subgroup performance differences coincide with differences in demographic representation and whether race-based sample weighting changes model performance.

### Research Question

**Does a machine-learning model for 30-day hospital readmission perform consistently across demographic groups, and how might differences in group representation affect model performance?**

### Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

The dataset contains hospital encounters for patients with diabetes and includes:

- Race
- Gender
- Age
- Diagnoses
- Medications
- Hospitalization characteristics
- Prior inpatient visits
- Prior outpatient visits
- Prior emergency visits
- Readmission outcome

The original dataset contains 101,766 hospital encounters involving 71,518 unique patients.

### Prediction Target

The original `readmitted` variable was converted into a binary outcome:

- `<30` = 1
- `>30` or `NO` = 0

The model therefore predicts whether a hospital encounter will be followed by readmission within 30 days.

### Methodology

**Data Preparation**

- Replaced `"?"` values with missing values.
- Removed variables with substantial missingness that were not required for the analysis.
- Simplified diagnosis codes into broader clinical categories.
- Encoded categorical predictors using one-hot encoding.
- Standardized numerical predictors.
- Excluded `encounter_id` and `patient_nbr` from model predictors.

**Patient-Level Train/Test Split**

Because individual patients may have multiple hospital encounters, the data were split using patient ID rather than randomly by encounter.

This prevents encounters from the same patient from appearing in both training and test datasets.

A `GroupShuffleSplit` was used with:

- 80% training
- 20% testing
- Random state = 42

No patients overlapped between the training and test sets.

### Models

Two models were evaluated:

1. Logistic Regression
2. XGBoost

Class imbalance was addressed using balanced class weights for Logistic Regression and `scale_pos
