# Databricks notebook source
# MAGIC %md
# MAGIC Read the Silver dataset from storage and build the four dimension tables for the Gold star schema.

# COMMAND ----------

from pyspark.sql.functions import regexp_extract, col

storage_account_name = "healthcarepipelinesa789"
silver_path = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/silver/diabetic_encounters"
gold_base = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/gold"

silver = spark.read.format("delta").load(silver_path)

dim_admission_type = silver.select("admission_type_id", "admission_type_desc").distinct().orderBy("admission_type_id")
dim_discharge_disposition = silver.select("discharge_disposition_id", "discharge_disposition_desc").distinct().orderBy("discharge_disposition_id")
dim_admission_source = silver.select("admission_source_id", "admission_source_desc").distinct().orderBy("admission_source_id")

dim_age_group = (
    silver.select("age").distinct()
    .withColumn("age_start", regexp_extract(col("age"), r"\[(\d+)-", 1).cast("int"))
    .withColumn("age_midpoint", col("age_start") + 5)
    .orderBy("age_start")
)

dim_admission_type.show(truncate=False)
dim_age_group.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Build the fact table: one row per encounter, with a binary readmission label and the numeric measures needed for analysis and modelling.

# COMMAND ----------

from pyspark.sql.functions import when as spark_when

fact_encounter = silver.withColumn(
    "readmitted_30d",
    spark_when(col("readmitted") == "<30", 1).otherwise(0)
).select(
    "encounter_id", "patient_nbr",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "age", "gender", "race",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "not_at_risk_of_readmission",
    "readmitted", "readmitted_30d"
)

print("Fact table rows:", fact_encounter.count())
fact_encounter.select("encounter_id", "readmitted", "readmitted_30d", "not_at_risk_of_readmission").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Verify the readmitted_30d label: confirm the `<30` case maps to 1, and check the full cross-tab for all three categories.

# COMMAND ----------

fact_encounter.filter(col("readmitted") == "<30").select(
    "encounter_id", "readmitted", "readmitted_30d"
).show(5)

fact_encounter.groupBy("readmitted", "readmitted_30d").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Write the fact table and four dimension tables to the gold folder in ADLS as Parquet, then read them back to verify.

# COMMAND ----------

fact_encounter.write.format("delta").mode("overwrite").save(f"{gold_base}/fact_encounter")
dim_admission_type.write.format("delta").mode("overwrite").save(f"{gold_base}/dim_admission_type")
dim_discharge_disposition.write.format("delta").mode("overwrite").save(f"{gold_base}/dim_discharge_disposition")
dim_admission_source.write.format("delta").mode("overwrite").save(f"{gold_base}/dim_admission_source")
dim_age_group.write.format("delta").mode("overwrite").save(f"{gold_base}/dim_age_group")

for name in ["fact_encounter", "dim_admission_type", "dim_discharge_disposition", "dim_admission_source", "dim_age_group"]:
    check = spark.read.format("delta").load(f"{gold_base}/{name}")
    print(name, "->", check.count(), "rows")