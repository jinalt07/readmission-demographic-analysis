import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Project setup works!")

pd.set_option("display.max_columns", None)


##================================================
## Load Data
##================================================

df = pd.read_csv("data/diabetic_data.csv")
print("\nShape")
print(df.shape)
print((df.head()))


##================================================
## Missing Data
##================================================

print("\nQuestion Mark Count")
question_mark_ct = (df == "?").sum()
print(question_mark_ct[question_mark_ct > 0].sort_values(ascending = False))

print("\nNaN Count")
nan_ct = df.isna().sum()
print(nan_ct[nan_ct > 0].sort_values(ascending= False))

## "?" to NaN Change
##==================================================

df.replace("?", np.nan, inplace= True)

print("\nMissing values as NaN")
missing_val = df.isna().sum()

print(missing_val[missing_val > 0].sort_values(ascending=False))

missing_val_percent = (df.isna().mean()*100).round(2)
print("\nMissing Value Percentage")
print(missing_val_percent[missing_val_percent > 0].sort_values(ascending=False))


##====================================================
## Data Cleaning
##====================================================

diabetes = df.drop(columns=["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]).copy()
print("\nShape")
print(df.shape)

print("\nShape Cleaned")
print(diabetes.shape)


##===================================================
## Coding Binary
##===================================================

diabetes["readmit_30"] = (diabetes["readmitted"] == "<30").astype(int)

print(diabetes["readmit_30"].value_counts())

## Checking duplicates
##===================================================

total_encounters = len(diabetes)

print("\nTotal patients:", total_encounters)

unique_patients = diabetes["patient_nbr"].nunique()
print("\nUnique patients:", unique_patients)

patient_counts = diabetes["patient_nbr"].value_counts()
print("\nPatient counts:", patient_counts)

repeated_patients = (patient_counts > 1).sum()
print("\nRepeated patients:", repeated_patients)

## Checking Demographics
##==================================================

print(diabetes["race"].value_counts(dropna=False))

print(diabetes["gender"].value_counts(dropna=False))

print(diabetes["age"].value_counts().sort_index())

# 1. Race

race_info = (diabetes.groupby("race", dropna= False)["readmit_30"].agg(total_encounters ="count", readmitted_30= "sum", readmission_rate= "mean"))

race_info["readmission_rate"] = (race_info["readmission_rate"]*100).round(2)

print("\n30-day readmission by race")
print(race_info)

# 2. Gender

gender_info = (diabetes.groupby("gender")["readmit_30"].agg(total_encounters= "count", readmitted_30 = "sum", readmission_rate= "mean"))

gender_info["readmission_rate"] = (gender_info["readmission_rate"]*100).round(2)

print("\n30-day readmission by gender")
print(gender_info)

# 3. Age

age_info = (diabetes.groupby("age")["readmit_30"].agg(total_encounters= "count", readmitted_30 = "sum", readmission_rate = "mean"))

age_info["readmission_rate"] = (age_info["readmission_rate"] *100).round(2)

print("\n30-day readmission by age")
print(age_info)

## Encounter Duplicates 
##================================================

encounters_duplicate = diabetes["encounter_id"].duplicated().sum()

print("\nEncounters duplicates", encounters_duplicate)

duplicate_rows = diabetes.duplicated().sum()

print("\nDuplicate rows", duplicate_rows)


##==============================================
## Prediction Variables Preparation
##==============================================

x = diabetes[["age", "gender", "race", "admission_type_id", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "diag_1", "diag_2", "diag_3", "insulin", "change", "diabetesMed"]].copy()
y = diabetes["readmit_30"].copy()

patients = diabetes["patient_nbr"].copy()

print("\nX Shape", x.shape)
print("\nY Shape", y.shape)
print("\nPatient Shape", patients.shape)

print(x.columns.tolist())

##==============================================
## Train and Test Split
##==============================================

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size= 0.20, random_state=42)

train_idx, test_idx = next(splitter.split(x, y, groups=patients))


x_train = x.iloc[train_idx].copy()
x_test = x.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

patients_train = patients.iloc[train_idx]
patients_test = patients.iloc[test_idx]

print("Train X", x_train.shape)
print("Test X", x_test.shape)

print("Train patients", patients_train.nunique())
print("Test patients", patients_test.nunique())

# Overlap

patient_overlap = set(patients_train).intersection(set(patients_test))
print("\nOverlap", len(patient_overlap))

# Average & Sum

print("\nTrain Admission", y_train.mean())
print("\nTest Admission", y_test.mean())

print("\nTrain Positives", y_train.sum())
print("\nTest Positives", y_test.sum())

##=================================================
## Missing Demographics
##================================================

print(x_train["race"].isna().sum())
print(x_train["gender"].isna().sum())
print(x_train["age"].isna().sum())
print(x_train["diag_1"].isna().sum())
print(x_train["diag_2"].isna().sum())
print(x_train["diag_3"].isna().sum())

print(x_train["gender"].value_counts(dropna=False))
print(x_train["age"].value_counts(dropna=False))

# Gender

print("\nClean Gender Train")
x_train["gender"] = x_train["gender"].replace("Unknown/Invalid", np.nan)
print(x_train["gender"].value_counts(dropna=False))
print("\nClean Gender Test") 
x_test["gender"] = x_test["gender"].replace("Unknown/Invalid", np.nan)
print(x_test["gender"].value_counts(dropna=False))

# Race

print(x_train["race"].isna().sum())
print(x_test["race"].isna().sum())

# Diagnosis

def category_diagnosis(code):
    if pd.isna(code):
        return np.nan
    code =str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    code = float(code)
    if 250<= code <251:
        return "Diabetes"
    elif 390 <= code <= 459 or code == 785:
        return "Circulatory"
    elif 460 <= code <= 519 or code == 786:
        return "Respiratory"
    elif 520 <= code <= 579 or code == 787:
        return "Digestive"
    elif 580 <= code <= 629 or code == 788:
        return "Genitourinary"
    elif 800 <= code <= 999:
        return "Injury"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 140 <= code <= 239:
        return "Neoplasms"
    else:
        return "Other"


x_train["diag_1_category"] = x_train["diag_1"].apply(category_diagnosis)
x_train["diag_2_category"] = x_train["diag_2"].apply(category_diagnosis)
x_train["diag_3_category"] = x_train["diag_3"].apply(category_diagnosis)

x_test["diag_1_category"] = x_test["diag_1"].apply(category_diagnosis)
x_test["diag_2_category"] = x_test["diag_2"].apply(category_diagnosis)
x_test["diag_3_category"] = x_test["diag_3"].apply(category_diagnosis)

print("diag1", x_train["diag_1_category"].value_counts(dropna=False))
print("diag2", x_train["diag_2_category"].value_counts(dropna=False))
print("diag3", x_train["diag_3_category"].value_counts(dropna=False))

x_train = x_train.drop(columns = ["diag_1", "diag_2", "diag_3"])
x_test = x_test.drop(columns = ["diag_1", "diag_2", "diag_3"])

print(x_train.columns)

##==============================================
## Cat and Num Features and Preparation
##===============================================

cat_features = ["age", "gender", "race", "admission_type_id", "insulin", "change", "diabetesMed", "diag_1_category", "diag_2_category", "diag_3_category"]

num_features = [ "time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]

print(x_train[cat_features].dtypes)
print(x_train[num_features].dtypes)

print(x_train[cat_features + num_features].isna().sum())


from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(handle_unknown= "ignore")


from sklearn.pipeline import Pipeline
cat_pipeline = Pipeline([("importer", cat_imputer), ("encoder", cat_encoder)])


from sklearn.preprocessing import StandardScaler
num_scaler = StandardScaler()


from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([("categorical", cat_pipeline, cat_features), ("numeric", num_scaler, num_features)])


##===================================================
## Data Analysis
##===================================================

# 1. Logistic Regression
##===================================================

from sklearn.linear_model import LogisticRegression

logistic_reg_model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression
                                                              (max_iter=2000,
                                                               class_weight="balanced",
                                                               random_state=42))])

logistic_reg_model.fit(x_train, y_train)
print("\nLogistic Regression Training Complete")

# Prediction and Probabilities on Test Data set

y_pred = logistic_reg_model.predict(x_test)
y_prob = logistic_reg_model.predict_proba(x_test)[:,1]

# Model Perfromance

from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score)

print("\nAUROC", roc_auc_score(y_test, y_prob))
print("\nAUPRC", average_precision_score(y_test, y_prob))
print("\nPrecision", precision_score(y_test, y_pred))
print("\nRecall", recall_score(y_test, y_pred))
print("\nF1", f1_score(y_test, y_pred))

# Confusion Matrix

from sklearn.metrics import confusion_matrix

conf_metrix = confusion_matrix(y_test, y_pred)

print(conf_metrix)


# Visualizations 

#1. CM

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Logistics Regression Confusion Matrix")
#plt.show()
plt.close()

#2 ROC Curve

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_prob)

plt.title("Logistic Regression ROC Curve")
#plt.show()
plt.close()

#3 P-R Curve

from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(y_test, y_prob)

plt.title("Logistic Regression P-R Curve")
#plt.show()
plt.close()


# 2. Gradient Boost Model
##========================================================

from xgboost import XGBClassifier

negatice_count = (y_train ==0).sum()
positive_count = (y_train == 1).sum()

scale_weight = negatice_count / positive_count
print("Scale Weight", scale_weight)

from xgboost import XGBClassifier

xgb_model = Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier(n_estimators=200, 
                                                                              max_depth= 3,
                                                                              learning_rate = 0.05,
                                                                              scale_pos_weight = scale_weight,
                                                                              random_state= 42,
                                                                              eval_metric= "logloss"))])

xgb_model.fit(x_train, y_train)
print("XGBoost Training Complete")

# Prediction and Probabilities on Test Data set

xgb_pred = xgb_model.predict(x_test)
xgb_prob = xgb_model.predict_proba(x_test)[:,1]

# Model Performance

print("\nAUROC:", roc_auc_score(y_test, xgb_prob))
print("\nAUPRC:", average_precision_score(y_test, xgb_prob))
print("\nPrecision:", precision_score(y_test, xgb_pred))
print("\nRecall:", recall_score(y_test, xgb_pred))
print("\nF1:", f1_score(y_test, xgb_pred))

# Confusion Matrix

xgb_conf = confusion_matrix(y_test, xgb_pred)

print(xgb_conf)

# Visualizations

#1. Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, xgb_pred)

plt.title("XGBoost Confusion Matrix")
#plt.show()
plt.close()

#2. ROC Curve

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test,xgb_prob)

plt.title("XGBoost ROC Curve")
#plt.show()
plt.close()

#3. P-R Curve

from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(y_test,xgb_prob)

plt.title("XGBoost Precision-Recall Curve")
#plt.show()
plt.close()

##===================================================
## Demographic Subgroup Analysis
##===================================================

# 1. Race
##===================================================

race_test = x_test[["race"]].copy()
race_test["readmitted_30"] = y_test.values

print(race_test.head())
print(race_test.columns)

print(race_test.groupby("race", dropna=False)["readmitted_30"].agg(["count", "sum", "mean"]))

race_test["prediction"] = xgb_pred
race_test["probability"] = xgb_prob

print(race_test.head())

# Caucasian Race

caucasian = race_test[race_test["race"] == "Caucasian"]

print("\nNum of encounters", len(caucasian))
print("Postive cases", caucasian["readmitted_30"].sum())

print("AUROC:",roc_auc_score(caucasian["readmitted_30"],caucasian["probability"]))
print("Precision:",precision_score(caucasian["readmitted_30"],caucasian["prediction"]))
print("Recall:",recall_score(caucasian["readmitted_30"],caucasian["prediction"]))
print("F1:",f1_score(caucasian["readmitted_30"],caucasian["prediction"]))

# African American

african_american = race_test[race_test["race"] == "AfricanAmerican"]

print("\nNum of encounters", len(african_american))
print("Positive cases", african_american["readmitted_30"].sum())

print("AUROC:",roc_auc_score(african_american["readmitted_30"],african_american["probability"]))
print("Precision:",precision_score(african_american["readmitted_30"],african_american["prediction"]))
print("Recall:",recall_score(african_american["readmitted_30"],african_american["prediction"]))
print("F1:",f1_score(african_american["readmitted_30"],african_american["prediction"]))

# Asian

asian= race_test[race_test["race"] == "Asian"]

print("\nNum of encounters", len(asian))
print("Positive cases", asian["readmitted_30"].sum())

print("AUROC:",roc_auc_score(asian["readmitted_30"],asian["probability"]))
print("Precision:",precision_score(asian["readmitted_30"],asian["prediction"]))
print("Recall:",recall_score(asian["readmitted_30"],asian["prediction"]))
print("F1:",f1_score(asian["readmitted_30"],asian["prediction"]))

# Hispanic

hispanic= race_test[race_test["race"] == "Hispanic"]

print("\nNum of encounters", len(hispanic))
print("Positive cases", hispanic["readmitted_30"].sum())

print("AUROC:",roc_auc_score(hispanic["readmitted_30"],hispanic["probability"]))
print("Precision:",precision_score(hispanic["readmitted_30"],hispanic["prediction"]))
print("Recall:",recall_score(hispanic["readmitted_30"],hispanic["prediction"]))
print("F1:",f1_score(hispanic["readmitted_30"],hispanic["prediction"]))

# Other

other= race_test[race_test["race"] == "Other"]

print("\nNum of encounters", len(other))
print("Positive cases", other["readmitted_30"].sum())

print("AUROC:",roc_auc_score(other["readmitted_30"],other["probability"]))
print("Precision:",precision_score(other["readmitted_30"],other["prediction"]))
print("Recall:",recall_score(other["readmitted_30"],other["prediction"]))
print("F1:",f1_score(other["readmitted_30"],other["prediction"]))

# Missing Values

missing= race_test[race_test["race"].isna()]

print("\nNum of encounters", len(missing))
print("Positive cases", missing["readmitted_30"].sum())

print("AUROC:",roc_auc_score(missing["readmitted_30"],missing["probability"]))
print("Precision:",precision_score(missing["readmitted_30"],missing["prediction"]))
print("Recall:",recall_score(missing["readmitted_30"],missing["prediction"]))
print("F1:",f1_score(missing["readmitted_30"],missing["prediction"]))

## Comparision Table

race_results = pd.DataFrame({"Race" : ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Missing"],
                            "Encounters": [len(caucasian), len(african_american), len(asian), len(hispanic), len(other), len(missing)],
                            "Positive_Cases": [caucasian["readmitted_30"].sum(), african_american["readmitted_30"].sum(), asian["readmitted_30"].sum(), hispanic["readmitted_30"].sum(), other["readmitted_30"].sum(), missing["readmitted_30"].sum()],
                            "AUROC": [roc_auc_score(caucasian["readmitted_30"], caucasian["probability"]),roc_auc_score(african_american["readmitted_30"], african_american["probability"]), roc_auc_score(asian["readmitted_30"], asian["probability"]),roc_auc_score(hispanic["readmitted_30"], hispanic["probability"]),roc_auc_score(other["readmitted_30"], other["probability"]),roc_auc_score(missing["readmitted_30"], missing["probability"])],
                            "Recall": [recall_score(caucasian["readmitted_30"], caucasian["prediction"]),recall_score(african_american["readmitted_30"], african_american["prediction"]),recall_score(asian["readmitted_30"], asian["prediction"]),recall_score(hispanic["readmitted_30"], hispanic["prediction"]),recall_score(other["readmitted_30"], other["prediction"]),recall_score(missing["readmitted_30"], missing["prediction"])],
                            "Precision": [precision_score(caucasian["readmitted_30"], caucasian["prediction"]),precision_score(african_american["readmitted_30"], african_american["prediction"]),precision_score(asian["readmitted_30"], asian["prediction"]),precision_score(hispanic["readmitted_30"], hispanic["prediction"]),precision_score(other["readmitted_30"], other["prediction"]),precision_score(missing["readmitted_30"], missing["prediction"])],
                            "F1": [f1_score(caucasian["readmitted_30"], caucasian["prediction"]),f1_score(african_american["readmitted_30"], african_american["prediction"]),f1_score(asian["readmitted_30"], asian["prediction"]),f1_score(hispanic["readmitted_30"], hispanic["prediction"]),f1_score(other["readmitted_30"], other["prediction"]),f1_score(missing["readmitted_30"], missing["prediction"])]})

print(race_results.round(3))

# 2. Gender
##===========================================

gender_test = x_test[["gender"]].copy()

gender_test["readmitted_30"] = y_test.values

print(gender_test.groupby("gender", dropna = False)["readmitted_30"].agg(["count", "sum", "mean"]))

gender_test["prediction"] = xgb_pred
gender_test["probability"] = xgb_prob

print(gender_test.head())

# Female

female = gender_test[gender_test["gender"] == "Female"]

print("\nNum of Encounter", len(female))
print("Positive cases", female["readmitted_30"].sum())

print("AUROC:",roc_auc_score(female["readmitted_30"],female["probability"]))
print("Precision:",precision_score(female["readmitted_30"],female["prediction"]))
print("Recall:",recall_score(female["readmitted_30"],female["prediction"]))
print("F1:",f1_score(female["readmitted_30"],female["prediction"]))

# Male

male = gender_test[gender_test["gender"] == "Male"]

print("\nNum of Encounter", len(male))
print("Positive cases", male["readmitted_30"].sum())

print("AUROC:",roc_auc_score(male["readmitted_30"],male["probability"]))
print("Precision:",precision_score(male["readmitted_30"],male["prediction"]))
print("Recall:",recall_score(male["readmitted_30"],male["prediction"]))
print("F1:",f1_score(male["readmitted_30"],male["prediction"]))

# Comparision Table

gender_results = pd.DataFrame({"Gender": ["Female", "Male"],
                               "Encounters": [len(female), len(male)],
                               "Positive_Cases": [female["readmitted_30"].sum(), male["readmitted_30"].sum()],
                               "AUROC": [roc_auc_score(female["readmitted_30"], female["probability"]), roc_auc_score(male["readmitted_30"], male["probability"])],
                               "Recall": [recall_score(female["readmitted_30"], female["prediction"]), recall_score(male["readmitted_30"], male["prediction"])],
                               "Precision": [precision_score(female["readmitted_30"], female["prediction"]),precision_score(male["readmitted_30"], male["prediction"])],
                               "F1": [f1_score(female["readmitted_30"], female["prediction"]),f1_score(male["readmitted_30"], male["prediction"])]})

print(gender_results.round(3))

# 3. Age
##=================================================

age_test = x_test[["age"]].copy()

age_test["readmitted_30"] = y_test.values

print(age_test.groupby("age", dropna= False)["readmitted_30"].agg(["count", "sum", "mean"]))

# Redefine Groups 

age_test["age_group"] = age_test["age"].replace({"[0-10)": "0-39","[10-20)": "0-39","[20-30)": "0-39","[30-40)": "0-39","[40-50)": "40-59","[50-60)": "40-59","[60-70)": "60-79","[70-80)": "60-79","[80-90)": "80-100","[90-100)": "80-100"})

print(age_test.groupby("age_group")["readmitted_30"].agg(["count", "sum", "mean"]))

age_test["prediction"] = xgb_pred

age_test["probability"] = xgb_prob

print(age_test.head())

# age 0-39

age_0_39 = age_test[age_test["age_group"] == "0-39"]

print("\nNum of Encounter", len(age_0_39))
print("Positive cases", age_0_39["readmitted_30"].sum())

print("AUROC:",roc_auc_score(age_0_39["readmitted_30"],age_0_39["probability"]))
print("Precision:",precision_score(age_0_39["readmitted_30"],age_0_39["prediction"]))
print("Recall:",recall_score(age_0_39["readmitted_30"],age_0_39["prediction"]))
print("F1:",f1_score(age_0_39["readmitted_30"],age_0_39["prediction"]))

# age 40-59

age_40_59 = age_test[age_test["age_group"] == "40-59"]

print("\nNum of Encounter", len(age_40_59))
print("Positive cases", age_40_59["readmitted_30"].sum())

print("AUROC:",roc_auc_score(age_40_59["readmitted_30"],age_40_59["probability"]))
print("Precision:",precision_score(age_40_59["readmitted_30"],age_40_59["prediction"]))
print("Recall:",recall_score(age_40_59["readmitted_30"],age_40_59["prediction"]))
print("F1:",f1_score(age_40_59["readmitted_30"],age_40_59["prediction"]))

# age 60-79

age_60_79 = age_test[age_test["age_group"] == "60-79"]

print("\nNum of Encounter", len(age_60_79))
print("Positive cases", age_60_79 ["readmitted_30"].sum())

print("AUROC:",roc_auc_score(age_60_79["readmitted_30"],age_60_79["probability"]))
print("Precision:",precision_score(age_60_79["readmitted_30"],age_60_79["prediction"]))
print("Recall:",recall_score(age_60_79["readmitted_30"],age_60_79["prediction"]))
print("F1:",f1_score(age_60_79["readmitted_30"],age_60_79["prediction"]))

# age 80-100

age_80_100 = age_test[age_test["age_group"] == "80-100"]

print("\nNum of Encounter", len(age_80_100))
print("Positive cases", age_80_100 ["readmitted_30"].sum())

print("AUROC:",roc_auc_score(age_80_100["readmitted_30"],age_80_100["probability"]))
print("Precision:",precision_score(age_80_100["readmitted_30"],age_80_100["prediction"]))
print("Recall:",recall_score(age_80_100["readmitted_30"],age_80_100["prediction"]))
print("F1:",f1_score(age_80_100["readmitted_30"],age_80_100["prediction"]))

# Comparision Table

age_results = pd.DataFrame({"Age_Group": ["0-39", "40-59", "60-79", "80-100"],
                            "Encounters": [len(age_0_39), len(age_40_59), len(age_60_79), len(age_80_100)],
                            "Positive_Cases": [age_0_39["readmitted_30"].sum(),age_40_59["readmitted_30"].sum(),age_60_79["readmitted_30"].sum(), age_80_100["readmitted_30"].sum()],
                            "AUROC": [roc_auc_score(age_0_39["readmitted_30"], age_0_39["probability"]),roc_auc_score(age_40_59["readmitted_30"], age_40_59["probability"]),roc_auc_score(age_60_79["readmitted_30"], age_60_79["probability"]),roc_auc_score(age_80_100["readmitted_30"], age_80_100["probability"])],
                            "Recall": [recall_score(age_0_39["readmitted_30"], age_0_39["prediction"]),recall_score(age_40_59["readmitted_30"], age_40_59["prediction"]),recall_score(age_60_79["readmitted_30"], age_60_79["prediction"]),recall_score(age_80_100["readmitted_30"], age_80_100["prediction"])],
                            "Precision": [precision_score(age_0_39["readmitted_30"], age_0_39["prediction"]),precision_score(age_40_59["readmitted_30"], age_40_59["prediction"]),precision_score(age_60_79["readmitted_30"], age_60_79["prediction"]),precision_score(age_80_100["readmitted_30"], age_80_100["prediction"])],
                            "F1": [f1_score(age_0_39["readmitted_30"], age_0_39["prediction"]),f1_score(age_40_59["readmitted_30"], age_40_59["prediction"]),f1_score(age_60_79["readmitted_30"], age_60_79["prediction"]),f1_score(age_80_100["readmitted_30"], age_80_100["prediction"])]})

print(age_results.round(3))

## Unique Patients Race

race_test["patient_nbr"] = patients_test.values
print(race_test.head())

print(race_test.groupby("race", dropna= False)["patient_nbr"].nunique())

caucasian = race_test[race_test["race"] == "Caucasian"]
african_american = race_test[race_test["race"] == "AfricanAmerican"]
asian = race_test[race_test["race"] == "Asian"]
hispanic = race_test[race_test["race"] == "Hispanic"]
other = race_test[race_test["race"] == "Other"]
missing= race_test[race_test["race"].isna()]

race_results["Unique_Patients"] = [
caucasian["patient_nbr"].nunique(),
african_american["patient_nbr"].nunique(),
asian["patient_nbr"].nunique(),
hispanic["patient_nbr"].nunique(),
other["patient_nbr"].nunique(),
missing["patient_nbr"].nunique()]

print(race_results.round(3))

## Unique Patients Gender

gender_test["patient_nbr"] = patients_test.values

print(gender_test.groupby("gender", dropna=False)["patient_nbr"].nunique())

gender_results["Unique_Patients"] = [gender_test[gender_test["gender"] == "Female"]["patient_nbr"].nunique(),
    gender_test[gender_test["gender"] == "Male"]["patient_nbr"].nunique()]

print(gender_results.round(3))

## Unique Patients Age

age_test["patient_nbr"] = patients_test.values

print(age_test.groupby("age_group")["patient_nbr"].nunique())

age_results["Unique_Patients"] = [age_test[age_test["age_group"] == "0-39"]["patient_nbr"].nunique(),
    age_test[age_test["age_group"] == "40-59"]["patient_nbr"].nunique(),
    age_test[age_test["age_group"] == "60-79"]["patient_nbr"].nunique(),
    age_test[age_test["age_group"] == "80-100"]["patient_nbr"].nunique()]

print(age_results.round(3))


##=============================================
## Representation Analysis
##=============================================

race_rep = race_results[["Race", "Unique_Patients"
, "Positive_Cases", "AUROC", "Recall"]]

print(race_rep)

race_codes = race_rep["Race"].astype("category")

# Recall Chart

plt.figure(figsize=(8,6))

scatter = plt.scatter(race_rep["Unique_Patients"], race_rep["Recall"], c=race_codes.cat.codes,cmap="tab10",s=100)

plt.xscale("log")

plt.xlabel("Number of Unique Patients")
plt.ylabel("Recall")
plt.title("Race Representation vs XGBoost Recall")

cbar = plt.colorbar(scatter, ticks=range(len(race_codes.cat.categories)))
cbar.ax.set_yticklabels(race_codes.cat.categories)
cbar.set_label("Race")

#plt.show()
plt.close()

# AUROC Chart

plt.figure(figsize=(8, 6))

scatter = plt.scatter(race_rep["Unique_Patients"],race_rep["AUROC"],c=race_codes.cat.codes,cmap="tab10",s=100)

plt.xscale("log")

plt.xlabel("Number of Unique Patients (log scale)")
plt.ylabel("AUROC")
plt.title("Race Representation vs XGBoost AUROC")

cbar = plt.colorbar(scatter, ticks=range(len(race_codes.cat.categories)))
cbar.ax.set_yticklabels(race_codes.cat.categories)
cbar.set_label("Race")

#plt.show()
plt.close()

##==================================================
## Representation Sensitivity Experiement
##===================================================

print(x_train["race"].value_counts(dropna=False))

# Sampled Weights

from sklearn.utils.class_weight import compute_sample_weight

race_for_weights = x_train["race"].fillna("Missing")

race_sample_weights = compute_sample_weight(class_weight="balanced", y=race_for_weights)

from sklearn.base import clone

weighted_xgb_model = clone(xgb_model)
weighted_xgb_model.fit( x_train, y_train,model__sample_weight=race_sample_weights)

print("Weighted XGBoost training complete")


weighted_xgb_pred = weighted_xgb_model.predict(x_test)
weighted_xgb_prob = weighted_xgb_model.predict_proba(x_test)[:, 1]

race_test["weighted_prediction"] = weighted_xgb_pred
race_test["weighted_probability"] = weighted_xgb_prob

print(race_test.head())

# Asian Group

asian = race_test[race_test["race"] == "Asian"]

print("AUROC:",roc_auc_score(asian["readmitted_30"],asian["weighted_probability"]))
print("Precision:",precision_score(asian["readmitted_30"],asian["weighted_prediction"]))
print("Recall:",recall_score(asian["readmitted_30"],asian["weighted_prediction"]))
print("F1:",f1_score(asian["readmitted_30"],asian["weighted_prediction"]))

# Caucasian Group

caucasian = race_test[race_test["race"] == "Caucasian"]

african_american = race_test[race_test["race"] == "AfricanAmerican"]

asian = race_test[race_test["race"] == "Asian"]

hispanic = race_test[race_test["race"] == "Hispanic"]

other = race_test[race_test["race"] == "Other"]

missing = race_test[race_test["race"].isna()]

race_results["Weighted_AUROC"] = [
    roc_auc_score(caucasian["readmitted_30"], caucasian["weighted_probability"]),
    roc_auc_score(african_american["readmitted_30"], african_american["weighted_probability"]),
    roc_auc_score(asian["readmitted_30"], asian["weighted_probability"]),
    roc_auc_score(hispanic["readmitted_30"], hispanic["weighted_probability"]),
    roc_auc_score(other["readmitted_30"], other["weighted_probability"]),
    roc_auc_score(missing["readmitted_30"], missing["weighted_probability"])]

race_results["Weighted_Recall"] = [
    recall_score(caucasian["readmitted_30"], caucasian["weighted_prediction"]),
    recall_score(african_american["readmitted_30"], african_american["weighted_prediction"]),
    recall_score(asian["readmitted_30"], asian["weighted_prediction"]),
    recall_score(hispanic["readmitted_30"], hispanic["weighted_prediction"]),
    recall_score(other["readmitted_30"], other["weighted_prediction"]),
    recall_score(missing["readmitted_30"], missing["weighted_prediction"])]

race_results["Weighted_Precision"] = [
    precision_score(caucasian["readmitted_30"], caucasian["weighted_prediction"]),
    precision_score(african_american["readmitted_30"], african_american["weighted_prediction"]),
    precision_score(asian["readmitted_30"], asian["weighted_prediction"]),
    precision_score(hispanic["readmitted_30"], hispanic["weighted_prediction"]),
    precision_score(other["readmitted_30"], other["weighted_prediction"]),
    precision_score(missing["readmitted_30"], missing["weighted_prediction"])]

race_results["Weighted_F1"] = [
    f1_score(caucasian["readmitted_30"], caucasian["weighted_prediction"]),
    f1_score(african_american["readmitted_30"], african_american["weighted_prediction"]),
    f1_score(asian["readmitted_30"], asian["weighted_prediction"]),
    f1_score(hispanic["readmitted_30"], hispanic["weighted_prediction"]),
    f1_score(other["readmitted_30"], other["weighted_prediction"]),
    f1_score(missing["readmitted_30"], missing["weighted_prediction"])]

print(race_results.round(3))

# XGB Model- impact of weighted

print("Weighted Overall AUROC:",roc_auc_score(y_test, weighted_xgb_prob))
print("Weighted Overall AUPRC:",average_precision_score(y_test, weighted_xgb_prob))
print("Weighted Overall Precision:",precision_score(y_test, weighted_xgb_pred))
print("Weighted Overall Recall:",recall_score(y_test, weighted_xgb_pred))
print("Weighted Overall F1:",f1_score(y_test, weighted_xgb_pred))

##==================================================
## Feature Importance
##=================================================

feature_names = xgb_model.named_steps["preprocessor"].get_feature_names_out()

print(feature_names[:20])
print("Total features:", len(feature_names))

feature_importance = xgb_model.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({"Feature": feature_names,"Importance": feature_importance})

importance_df = importance_df.sort_values(by="Importance",ascending=False)

print(importance_df.head(15))


top_features = importance_df.head(15).copy()

top_features["Feature"] = (top_features["Feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False))

plt.figure(figsize=(10, 7))

plt.barh(top_features["Feature"],top_features["Importance"])

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importances")

plt.tight_layout()
#plt.show()
plt.close()


##======================================
## Saving Outputs
##=========================================

from pathlib import Path

project_diabetes = Path(__file__). resolve().parents[1]

figures = project_diabetes / "figures"
results = project_diabetes / "results"

figures.mkdir(exist_ok=True)
results.mkdir(exist_ok=True)

print(figures)
print(results)

##============================================
## Final Report Charts
##=============================================


## Subgroup Performance - Recall

plt.figure(figsize=(9,6))

plt.bar(race_results["Race"], race_results["Recall"])

plt.xlabel("Race")
plt.ylabel("Recall")
plt.title("XGBoost Recall by Race")

plt.xticks(rotation=30)
plt.ylim(0, 0.7)

plt.tight_layout()
#plt.show()

plt.savefig(figures / "race_recall.png", dpi=300, bbox_inches = "tight")
plt.close()

## Model Comparision

model_compare = pd.DataFrame({"Model" : ["Logistic Regression", "XGBoost"],
                              "AUROC": [roc_auc_score(y_test, y_prob), roc_auc_score(y_test, xgb_prob)],
                              "AUPRC" : [average_precision_score(y_test, y_prob), average_precision_score(y_test, xgb_prob)],
                              "Precision" : [precision_score(y_test, y_pred), precision_score(y_test, xgb_pred)],
                              "Recall" : [recall_score(y_test, y_pred), recall_score(y_test, xgb_pred)],
                              "F1": [f1_score(y_test, y_pred), f1_score(y_test, xgb_pred)]})

print(model_compare.round(3))

model_compare.to_csv(results / "model_compare_results.csv", index=False)

## Demographic Subgroup Save

race_results.to_csv(results / "race_subgroup_results.csv", index= False)

gender_results.to_csv(results / "gender_subgroup_results.csv", index=False)

age_results.to_csv(results / "age_subgroup_results.csv", index=False)


## ROC Curve Compare - Figure

from sklearn.metrics import RocCurveDisplay

fig, ax= plt.subplots(figsize=(8,6))

RocCurveDisplay.from_predictions(y_test, y_prob, name= "Logistic Regression", ax=ax)
RocCurveDisplay.from_predictions(y_test, xgb_prob, name= "XGBoost", ax=ax)

plt.title("ROC Curve: Logistic Regression vs XGBoost")

plt.tight_layout()


plt.savefig(figures / "model_roc_compare.png", dpi=300, bbox_inches= "tight")
#plt.show()
plt.close()

## Precision- Recall Curve Compare - Figure

from sklearn.metrics import PrecisionRecallDisplay

fig, ax = plt.subplots(figsize= (8,6))

PrecisionRecallDisplay.from_predictions(y_test, y_prob, name= "Logistic Regression", ax=ax)

PrecisionRecallDisplay.from_predictions(y_test, xgb_prob, name= "XGBoost", ax=ax)

plt.title("Precision-Recall Curve: Logistic Regression vs XGBoost")

plt.tight_layout()

plt.savefig(figures / "model_precision_recall_compare.png", dpi=300, bbox_inches= "tight")
#plt.show()
plt.close()


## Race Representation - Figure

plt.figure(figsize=(9,6))
plt.bar(race_results["Race"], race_results["Unique_Patients"])

plt.xlabel("Race")
plt.ylabel("No. of Unique Patients")
plt.title("Patient Representation by Race")

plt.tight_layout()

plt.savefig(figures / "race_representation.png", dpi= 300, bbox_inches= "tight")
plt.close()

## Race Readmission & Figure

race_readmission = (race_test.groupby("race", dropna= False)["readmitted_30"].mean().reset_index())

race_readmission["race"] = race_readmission["race"].fillna("Missing")

race_readmission["readmission_rate"] = race_readmission["readmitted_30"] *100
print(race_readmission)

plt.figure(figsize=(9,6))

plt.bar(race_readmission["race"], race_readmission["readmission_rate"])

plt.xlabel("Race")
plt.ylabel("30-Day Readmission Rate")
plt.title("30-Day Readmission Rate by Race")

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig( figures / "race_readmission_rate.png",dpi=300,bbox_inches="tight")
plt.close()

## XGBoost confusion matrix

gig, ax= plt.subplots(figsize=(7,6))

ConfusionMatrixDisplay.from_predictions(y_test, xgb_pred, ax=ax)

plt.title("XGBoost Confusion Matrix")
plt.tight_layout()

plt.savefig(figures/ "xgboost_confusion_matrix.png", dpi= 300, bbox_inches= "tight")
plt.close()

## Race Representation vs Recall

race_codes = race_rep["Race"].astype("category")

plt.figure(figsize=(9,6))

scatter = plt.scatter(race_rep["Unique_Patients"], race_rep["Recall"], c=race_codes.cat.codes, cmap= "tab10", s=100)

plt.xscale("log")
plt.xlabel("Number of Unique Patients (Log Scale)")
plt.ylabel("Recall")
plt.title("Race Representation vs XGBoost Recall")

cbar= plt.colorbar(scatter, ticks=range(len(race_codes.cat.categories)))

cbar.ax.set_yticklabels(race_codes.cat.categories)

cbar.set_label("Race")
plt.tight_layout()

plt.savefig(figures / "race_representation_vs_recall.png", dpi=300, bbox_inches= "tight")
plt.close()


## Recall by Gender

plt.figure(figsize=(7,5))

plt.bar(gender_results["Gender"], gender_results["Recall"])

plt.xlabel("Gender")
plt.ylabel("Recall")
plt.title("XGBoost Recall by Gender")

plt.ylim(0, 0.7)
plt.tight_layout()

plt.savefig(figures / "gender_recall.png", dpi=300, bbox_inches= "tight")
plt.close()

## Recall by Age Group

plt.figure(figsize=(7,5))

plt.bar(age_results["Age_Group"], age_results["Recall"])

plt.xlabel("Age Group")
plt.ylabel("Recall")
plt.title("XGBoost Recall by Age Group")

plt.ylim(0, 0.7)

plt.tight_layout()

plt.savefig(figures / "age_recall.png", dpi=300, bbox_inches = "tight")
plt.close()

## AUROC by Age Group

plt.figure(figsize=(7,5))

plt.bar(age_results["Age_Group"], age_results["AUROC"])

plt.xlabel("Age Group")
plt.ylabel("AUROC")
plt.title("XGBoost AUROC by Age Group")

plt.ylim(0.5, 0.75)

plt.tight_layout()

plt.savefig( figures / "age_auroc.png", dpi=300, bbox_inches = "tight")
plt.close()

## Weighted before vs After

x_positions = np.arange(len(race_results["Race"]))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(x_positions - width/2, race_results["Recall"], width, label= "Original XGBoost")

plt.bar(x_positions + width / 2, race_results["Weighted_Recall"], width, label= "Race Weighted XGBoost")

plt.xticks(x_positions, race_results["Race"], rotation=30)

plt.xlabel("Race")
plt.ylabel("Recall")
plt.title("Race Subgroup Recall Before and After Representation Weighting")

plt.legend()
plt.tight_layout()
plt.savefig( figures / "race_weighting_recall_comparison.png", dpi= 300, bbox_inches= "tight")

## Weighted Overall Results - Figure

weighting_results = pd.DataFrame({"Model": ["Original XGBoost", "Race Weighted XGBoost"],
                                  "AUROC": [roc_auc_score(y_test, xgb_prob), roc_auc_score(y_test, weighted_xgb_prob)],
                                  "AUPRC": [average_precision_score(y_test, xgb_prob), average_precision_score(y_test, weighted_xgb_prob)],
                                  "Precision": [precision_score(y_test, xgb_pred), precision_score(y_test, weighted_xgb_pred)],
                                  "Recall": [recall_score(y_test, xgb_pred), recall_score(y_test, weighted_xgb_pred)],
                                  "F1": [f1_score(y_test, xgb_pred), f1_score(y_test, weighted_xgb_pred)]})

print(weighting_results.round(3))

weighting_results.to_csv(results / "weighting_sensitivity_results.csv", index=False)
plt.close()

## Feature Importance Save

plt.figure(figsize=(10, 7))

plt.barh(top_features["Feature"],top_features["Importance"])

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importances")

plt.tight_layout()

plt.savefig( figures / "xgboost_feature_importance.png", dpi=300, bbox_inches= "tight")

importance_df.to_csv(results / "xgboost_feature_importance.csv", index=False)
plt.close()
