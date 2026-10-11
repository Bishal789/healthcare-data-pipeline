# Databricks notebook source
# MAGIC %md
# MAGIC Read the Gold fact table, keep only patients who could be readmitted, and check class balance and patient repetition before modelling.

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct

base = "abfss://healthcare-data@healthcarepipelinesa789.dfs.core.windows.net"
fact = spark.read.format("delta").load(f"{base}/gold/fact_encounter")

model_df = fact.filter(col("not_at_risk_of_readmission") == False)

total = model_df.count()
positives = model_df.filter(col("readmitted_30d") == 1).count()
patients = model_df.select(countDistinct("patient_nbr")).collect()[0][0]

print("Rows for modelling:", total)
print("Readmitted within 30 days:", positives, f"({positives/total:.1%})")
print("Distinct patients:", patients, "| encounters per patient:", round(total / patients, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC Select features (nothing that reveals the outcome), convert to pandas, and split into train and test sets by patient so no patient appears in both.

# COMMAND ----------

from pyspark.sql.functions import col, regexp_extract
from sklearn.model_selection import GroupShuffleSplit

num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
cat_cols = ["gender", "race", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
            "admission_type_id", "discharge_disposition_id", "admission_source_id"]

model_sdf = model_df.withColumn(
    "age_midpoint", regexp_extract(col("age"), r"\[(\d+)-", 1).cast("int") + 5
)

pdf = model_sdf.select(["patient_nbr", "readmitted_30d", "age_midpoint"] + num_cols + cat_cols).toPandas()
for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    pdf[c] = pdf[c].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(pdf, groups=pdf["patient_nbr"]))
train, test = pdf.iloc[train_idx], pdf.iloc[test_idx]

print("Train rows:", len(train), "| Test rows:", len(test))
print("Patients in both sets:", len(set(train["patient_nbr"]) & set(test["patient_nbr"])))
print("Readmission rate - train: %.1f%% | test: %.1f%%" % (
    train["readmitted_30d"].mean() * 100, test["readmitted_30d"].mean() * 100))

# COMMAND ----------

# MAGIC %md
# MAGIC Train a logistic regression baseline inside a preprocessing pipeline and evaluate it on the held-out patients with ROC-AUC and PR-AUC.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

features_num = num_cols + ["age_midpoint"]
features_cat = cat_cols

X_train, y_train = train[features_num + features_cat], train["readmitted_30d"]
X_test, y_test = test[features_num + features_cat], test["readmitted_30d"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), features_num),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), features_cat),
])

logreg = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
logreg.fit(X_train, y_train)

proba = logreg.predict_proba(X_test)[:, 1]
print("ROC-AUC: %.3f" % roc_auc_score(y_test, proba))
print("PR-AUC:  %.3f  (random guessing would score about %.3f)" % (
    average_precision_score(y_test, proba), y_test.mean()))

# COMMAND ----------

# MAGIC %md
# MAGIC Train a gradient boosting model on the same split and compare it with the logistic regression baseline.

# COMMAND ----------

from sklearn.ensemble import HistGradientBoostingClassifier

preprocess_gb = ColumnTransformer([
    ("num", "passthrough", features_num),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), features_cat),
])

gboost = Pipeline([
    ("prep", preprocess_gb),
    ("model", HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_depth=4,
        class_weight="balanced", early_stopping=False, random_state=42)),
])
gboost.fit(X_train, y_train)

proba_gb = gboost.predict_proba(X_test)[:, 1]
print("Gradient boosting  ROC-AUC: %.3f | PR-AUC: %.3f" % (
    roc_auc_score(y_test, proba_gb), average_precision_score(y_test, proba_gb)))
print("Logistic regression ROC-AUC: %.3f | PR-AUC: %.3f" % (
    roc_auc_score(y_test, proba), average_precision_score(y_test, proba)))

# COMMAND ----------

# MAGIC %md
# MAGIC Inspect the logistic regression coefficients to see which features drive risk, and measure how well the highest-scored 10% of patients capture real readmissions.

# COMMAND ----------

import numpy as np
import pandas as pd

# 1. Which features push risk up or down
names = logreg.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(logreg.named_steps["model"].coef_[0], index=names).sort_values()
print("Strongest risk REDUCERS:")
print(coefs.head(8).round(3).to_string())
print("\nStrongest risk INCREASERS:")
print(coefs.tail(8).round(3).to_string())

# 2. Practical value: how many readmissions are in the top 10% of scores?
res = pd.DataFrame({"y": y_test.values, "p": proba})
res["decile"] = pd.qcut(res["p"].rank(method="first"), 10, labels=False)  # 9 = highest risk
top = res[res["decile"] == 9]
print("\nReadmission rate in top 10%% of scores: %.1f%% (overall: %.1f%%)" % (
    top["y"].mean() * 100, res["y"].mean() * 100))
print("Share of all readmissions captured in top 10%%: %.1f%%" % (
    top["y"].sum() / res["y"].sum() * 100))

# COMMAND ----------

# MAGIC %md
# MAGIC Produce out-of-fold risk scores for every at-risk patient (grouped by patient), assign risk bands, and write the results to Gold as a Delta table.

# COMMAND ----------

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict

full = model_sdf.select(
    ["encounter_id", "patient_nbr", "readmitted_30d", "age_midpoint"] + num_cols + cat_cols
).toPandas()
for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    full[c] = full[c].astype(str)

full["risk_score"] = cross_val_predict(
    logreg, full[features_num + features_cat], full["readmitted_30d"],
    groups=full["patient_nbr"], cv=GroupKFold(n_splits=5), method="predict_proba"
)[:, 1]

rank = full["risk_score"].rank(pct=True)
full["risk_band"] = np.where(rank > 0.90, "High", np.where(rank > 0.70, "Medium", "Low"))

print(full.groupby("risk_band")["readmitted_30d"].agg(["count", "mean"]).round(3))
print("Out-of-fold ROC-AUC: %.3f" % roc_auc_score(full["readmitted_30d"], full["risk_score"]))

out = spark.createDataFrame(full[["encounter_id", "risk_score", "risk_band"]])
out.write.format("delta").mode("overwrite").save(f"{base}/gold/readmission_risk")
print("Written:", spark.read.format("delta").load(f"{base}/gold/readmission_risk").count(), "rows")