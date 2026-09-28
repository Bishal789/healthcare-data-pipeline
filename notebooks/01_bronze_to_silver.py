# Databricks notebook source
# MAGIC %md
# MAGIC Read raw Bronze data from ADLS to confirm the pipeline connection and inspect its structure.

# COMMAND ----------

from pyspark.sql.functions import col, when, count, coalesce, lit

storage_account_name = "healthcarepipelinesa789"
bronze_path = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/bronze/diabetic_data.csv"

df = spark.read.csv(bronze_path, header=True, inferSchema=True)
print("Row count:", df.count())
print("Column count:", len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC Count `"?"` placeholders per string column to see exactly where they occur.

# COMMAND ----------

string_columns = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]

df.select([
    count(when(col(c) == "?", c)).alias(c) for c in string_columns
]).show(vertical=True, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Replace `"?"` with true nulls, on string columns only.

# COMMAND ----------

df_clean = df
for column in string_columns:
    df_clean = df_clean.withColumn(
        column,
        when(col(column) == "?", None).otherwise(col(column))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Verify no `"?"` values remain in any string column.

# COMMAND ----------

df_clean.select([
    count(when(col(c) == "?", c)).alias(c) for c in string_columns
]).show(vertical=True, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Apply the missing-value treatments: drop `weight`, recode lab/admin columns, drop rows with missing diagnoses.

# COMMAND ----------

df_silver = df_clean.drop("weight")

for column in ["max_glu_serum", "A1Cresult"]:
    df_silver = df_silver.withColumn(column, coalesce(col(column), lit("Not Tested")))

for column in ["medical_specialty", "payer_code", "race"]:
    df_silver = df_silver.withColumn(column, coalesce(col(column), lit("Unknown")))

for column in ["diag_2", "diag_3"]:
    df_silver = df_silver.withColumn(column, coalesce(col(column), lit("None")))

df_silver = df_silver.dropna(subset=["diag_1"])

print("Row count after cleaning:", df_silver.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Read the raw ID-mapping file as plain text, since it holds three tables stacked in one file.

# COMMAND ----------

mapping_path = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/bronze/IDS_mapping.csv"
raw_mapping = spark.read.text(mapping_path)
print("Lines read:", raw_mapping.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Parse the mapping file into three separate lookup sections.

# COMMAND ----------

lines = [row.value for row in raw_mapping.collect()]

sections = {}
current_key = None
current_rows = []

for line in lines:
    line = line.strip()
    if line == "" or line == ",":
        if current_key:
            sections[current_key] = current_rows
        current_key = None
        current_rows = []
        continue
    if line.endswith(",description"):
        current_key = line.split(",")[0]
        current_rows = []
        continue
    parts = line.split(",", 1)
    if len(parts) == 2:
        current_rows.append((parts[0].strip(), parts[1].strip()))

if current_key:
    sections[current_key] = current_rows

for key, rows in sections.items():
    print(key, "->", len(rows), "entries")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC Convert each lookup section into a small Spark DataFrame with an integer ID column.

# COMMAND ----------

admission_type_df = spark.createDataFrame(
    sections["admission_type_id"], ["admission_type_id", "admission_type_desc"]
).withColumn("admission_type_id", col("admission_type_id").cast("int"))

discharge_disposition_df = spark.createDataFrame(
    sections["discharge_disposition_id"], ["discharge_disposition_id", "discharge_disposition_desc"]
).withColumn("discharge_disposition_id", col("discharge_disposition_id").cast("int"))

admission_source_df = spark.createDataFrame(
    sections["admission_source_id"], ["admission_source_id", "admission_source_desc"]
).withColumn("admission_source_id", col("admission_source_id").cast("int"))

print(admission_type_df.count(), discharge_disposition_df.count(), admission_source_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC Join the lookups onto the main data and confirm no rows were lost or duplicated.

# COMMAND ----------

rows_before = df_silver.count()

df_silver = df_silver.join(admission_type_df, on="admission_type_id", how="left")
df_silver = df_silver.join(discharge_disposition_df, on="discharge_disposition_id", how="left")
df_silver = df_silver.join(admission_source_df, on="admission_source_id", how="left")

print("Rows before joins:", rows_before)
print("Rows after joins: ", df_silver.count())

df_silver.select(
    "admission_type_id", "admission_type_desc",
    "discharge_disposition_id", "discharge_disposition_desc",
    "admission_source_id", "admission_source_desc"
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Check for duplicate encounters, since each `encounter_id` should appear exactly once.

# COMMAND ----------

total_rows = df_silver.count()
distinct_encounters = df_silver.select("encounter_id").distinct().count()
distinct_full_rows = df_silver.dropDuplicates().count()

print("Total rows:              ", total_rows)
print("Distinct encounter_ids:  ", distinct_encounters)
print("Distinct full rows:      ", distinct_full_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC Profile the distinct values in key categorical columns to find invalid or unexpected entries before writing validation rules.

# COMMAND ----------

for c in ["gender", "age", "readmitted", "change", "diabetesMed"]:
    print(f"--- {c} ---")
    df_silver.groupBy(c).count().orderBy(c).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Quarantine rows with invalid gender values, then remove them from the Silver dataset.

# COMMAND ----------

invalid_gender = df_silver.filter(col("gender") == "Unknown/Invalid")
print("Quarantined rows:", invalid_gender.count())
invalid_gender.select("encounter_id", "patient_nbr", "gender", "age", "readmitted").show()

df_silver = df_silver.filter(col("gender") != "Unknown/Invalid")
print("Silver rows after removal:", df_silver.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Standardise placeholder descriptions ("NULL", "Not Available", "Not Mapped", "Unknown/Invalid") into a single "Unknown" label.

# COMMAND ----------

placeholder_labels = ["NULL", "Not Available", "Not Mapped", "Unknown/Invalid"]

for desc_col in ["admission_type_desc", "discharge_disposition_desc", "admission_source_desc"]:
    df_silver = df_silver.withColumn(
        desc_col,
        when(col(desc_col).isin(placeholder_labels), "Unknown")
        .otherwise(coalesce(col(desc_col), lit("Unknown")))
    )

for desc_col in ["admission_type_desc", "discharge_disposition_desc", "admission_source_desc"]:
    print(f"--- {desc_col} ---")
    df_silver.groupBy(desc_col).count().orderBy(col("count").desc()).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Add a business-rule flag for patients who died or went to hospice, who cannot realistically be readmitted.

# COMMAND ----------

not_at_risk_codes = [11, 13, 14, 19, 20, 21]

df_silver = df_silver.withColumn(
    "not_at_risk_of_readmission",
    col("discharge_disposition_id").isin(not_at_risk_codes)
)

df_silver.groupBy("not_at_risk_of_readmission").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Summarise the numeric columns (count, min, mean, max) to catch negative, impossible or suspicious values.

# COMMAND ----------

numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"
]

df_silver.select(numeric_cols).summary("count", "min", "mean", "max").show()

# COMMAND ----------

# MAGIC %md
# MAGIC the cleaned Silver dataset and the quarantined rows to the silver folder in ADLS as Parquet, then read it back to verify.

# COMMAND ----------

silver_path = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/silver/diabetic_encounters"
quarantine_path = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/silver/quarantine_invalid_gender"

df_silver.write.mode("overwrite").parquet(silver_path)
invalid_gender.write.mode("overwrite").parquet(quarantine_path)

# Read back from storage to prove the write worked
silver_check = spark.read.parquet(silver_path)
quarantine_check = spark.read.parquet(quarantine_path)

print("Silver rows:", silver_check.count(), "| columns:", len(silver_check.columns))
print("Quarantine rows:", quarantine_check.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Run all data quality rules against the Silver dataset and produce a pass/fail report.

# COMMAND ----------

silver = spark.read.parquet(silver_path)
total = silver.count()

checks = []

def add_check(name, category, failing_rows):
    checks.append((name, category, failing_rows, total, "PASS" if failing_rows == 0 else "FAIL"))

# Uniqueness
add_check("encounter_id is unique", "Duplicates",
          total - silver.select("encounter_id").distinct().count())

# Completeness: key columns must not be null
for c in ["encounter_id", "patient_nbr", "diag_1", "readmitted", "gender", "age"]:
    add_check(f"{c} is not null", "Completeness", silver.filter(col(c).isNull()).count())

# Valid values
add_check("readmitted is NO, <30 or >30", "Validity",
          silver.filter(~col("readmitted").isin("NO", "<30", ">30")).count())
add_check("gender is Male or Female", "Validity",
          silver.filter(~col("gender").isin("Male", "Female")).count())

# Numeric ranges
add_check("time_in_hospital between 1 and 14", "Range",
          silver.filter(~col("time_in_hospital").between(1, 14)).count())
for c in ["num_lab_procedures", "num_procedures", "num_medications",
          "number_outpatient", "number_emergency", "number_inpatient"]:
    add_check(f"{c} is not negative", "Range", silver.filter(col(c) < 0).count())

# Referential integrity: every ID code has a real description
for c in ["admission_type_desc", "discharge_disposition_desc", "admission_source_desc"]:
    add_check(f"{c} is not null", "Referential integrity", silver.filter(col(c).isNull()).count())

# Business rules
add_check("number_diagnoses is at least 1", "Business rule",
          silver.filter(col("number_diagnoses") < 1).count())
add_check("no '?' placeholders remain", "Validity",
          sum(silver.filter(col(c) == "?").count() for c in silver.columns
              if dict(silver.dtypes)[c] == "string"))

report = spark.createDataFrame(
    checks, ["check_name", "category", "failing_rows", "total_rows", "status"])
report.show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Add a consistency rule linking diagnosis codes to number_diagnoses, then save the data quality report to the silver folder.

# COMMAND ----------

inconsistent = silver.filter(
    ((col("number_diagnoses") >= 2) & (col("diag_2") == "None")) |
    ((col("number_diagnoses") >= 3) & (col("diag_3") == "None"))
).count()
add_check("diagnosis codes match number_diagnoses", "Business rule", inconsistent)

report = spark.createDataFrame(
    checks, ["check_name", "category", "failing_rows", "total_rows", "status"])

report.filter(col("status") == "FAIL").show(truncate=False)

report_path = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/silver/data_quality_report"
report.write.mode("overwrite").parquet(report_path)
print("Total checks:", report.count(), "| Failed:", report.filter(col("status") == "FAIL").count())