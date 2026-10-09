# Databricks notebook source
# MAGIC %md
# MAGIC **Demonstration notebook. Every step was run once on 9 Oct 2026 and all tables have since been converted to Delta. Do not re-run.**

# COMMAND ----------

dbutils.notebook.exit("Demonstration notebook, already run. Do not re-run.")

# COMMAND ----------

# MAGIC %md
# MAGIC Convert fact_encounter from Parquet to Delta format, then confirm it reads back correctly.

# COMMAND ----------

storage_account_name = "healthcarepipelinesa789"
gold_base = f"abfss://healthcare-data@{storage_account_name}.dfs.core.windows.net/gold"

# Read the existing Parquet table
fact_encounter = spark.read.parquet(f"{gold_base}/fact_encounter")

# Write it out again, in Delta format, to a new path
fact_encounter.write.format("delta").mode("overwrite").save(f"{gold_base}/fact_encounter_delta")

# Read it back as Delta and confirm
fact_delta = spark.read.format("delta").load(f"{gold_base}/fact_encounter_delta")
print("Row count:", fact_delta.count())

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC Attempt a schema-mismatched write to the Delta table, to see schema enforcement reject it rather than silently corrupting the data.

# COMMAND ----------

from pyspark.sql.functions import lit

# Deliberately corrupt the schema: turn time_in_hospital into a string
bad_data = fact_delta.withColumn("time_in_hospital", lit("not_a_number"))

# Try to append it to the existing Delta table, no special flags
bad_data.write.format("delta").mode("append").save(f"{gold_base}/fact_encounter_delta")

# COMMAND ----------

bad_data.write.mode("append").parquet(f"{gold_base}/fact_encounter")
print("Parquet write succeeded — no error raised")

# COMMAND ----------

# fact_encounter.write.mode("overwrite").parquet(f"{gold_base}/fact_encounter")
# print("Restored:", spark.read.parquet(f"{gold_base}/fact_encounter").count())

# COMMAND ----------

# MAGIC %md
# MAGIC Check the Delta table's version history, then query an earlier version using time travel.

# COMMAND ----------

from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, f"{gold_base}/fact_encounter_delta")
history = delta_table.history()
history.select("version", "timestamp", "operation").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Use time travel to query version 0 of the Delta table and confirm it's retrievable.

# COMMAND ----------

fact_v0 = spark.read.format("delta").option("versionAsOf", 0).load(f"{gold_base}/fact_encounter_delta")
fact_v1 = spark.read.format("delta").option("versionAsOf", 1).load(f"{gold_base}/fact_encounter_delta")

print("Version 0 row count:", fact_v0.count())
print("Version 1 row count:", fact_v1.count())

# COMMAND ----------

fact_v1 = spark.read.format("delta").option("versionAsOf", 1).load(f"{gold_base}/fact_encounter_delta")
print("Version 1 row count:", fact_v1.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Demonstrate schema evolution on a scratch copy: a new column is rejected by default, then accepted when mergeSchema is explicitly enabled.

# COMMAND ----------

from pyspark.sql.functions import when, col

gold_base = "abfss://healthcare-data@healthcarepipelinesa789.dfs.core.windows.net/gold"
scratch_path = f"{gold_base}/scratch_schema_evolution"

# Scratch copy of the Delta table
spark.read.format("delta").load(f"{gold_base}/fact_encounter_delta") \
    .write.format("delta").mode("overwrite").save(scratch_path)

# A small batch of 5 rows carrying one extra column
new_batch = (
    spark.read.format("delta").load(scratch_path).limit(5)
    .withColumn("long_stay", when(col("time_in_hospital") >= 7, 1).otherwise(0))
)

# Attempt 1: no flag, so Delta should reject the new column
try:
    new_batch.write.format("delta").mode("append").save(scratch_path)
    print("Append succeeded (unexpected)")
except Exception as e:
    print("Rejected:", str(e)[:150])

# Attempt 2: explicitly allow the schema to evolve
new_batch.write.format("delta").mode("append").option("mergeSchema", "true").save(scratch_path)

result = spark.read.format("delta").load(scratch_path)
print("Rows:", result.count(), "| Columns:", len(result.columns))
result.groupBy("long_stay").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the Silver table and the four Gold dimensions from Parquet to Delta, verify row counts match, and remove the scratch table.

# COMMAND ----------

base = "abfss://healthcare-data@healthcarepipelinesa789.dfs.core.windows.net"

to_convert = [
    ("silver/diabetic_encounters",           "silver/diabetic_encounters_delta"),
    ("gold/dim_admission_type",              "gold/dim_admission_type_delta"),
    ("gold/dim_discharge_disposition",       "gold/dim_discharge_disposition_delta"),
    ("gold/dim_admission_source",            "gold/dim_admission_source_delta"),
    ("gold/dim_age_group",                   "gold/dim_age_group_delta"),
]

for src, dst in to_convert:
    spark.read.parquet(f"{base}/{src}").write.format("delta").mode("overwrite").save(f"{base}/{dst}")
    n_src = spark.read.parquet(f"{base}/{src}").count()
    n_dst = spark.read.format("delta").load(f"{base}/{dst}").count()
    print(f"{dst}: parquet {n_src} -> delta {n_dst}", "OK" if n_src == n_dst else "MISMATCH")

# Remove the scratch table from the schema evolution demo
dbutils.fs.mv(f"{base}/gold/scratch_schema_evolution", True)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the quarantine table and the data quality report from Parquet to Delta.

# COMMAND ----------

base = "abfss://healthcare-data@healthcarepipelinesa789.dfs.core.windows.net"

for t in ["silver/quarantine_invalid_gender", "silver/data_quality_report"]:
    spark.read.parquet(f"{base}/{t}").write.format("delta").mode("overwrite").save(f"{base}/{t}_delta")
    print(t, "->", spark.read.format("delta").load(f"{base}/{t}_delta").count(), "rows")

# COMMAND ----------

# MAGIC %md
# MAGIC Swap each verified Delta copy into the original path, replacing the Parquet version.

# COMMAND ----------

tables = [
    "silver/diabetic_encounters", "silver/quarantine_invalid_gender", "silver/data_quality_report",
    "gold/fact_encounter", "gold/dim_admission_type", "gold/dim_discharge_disposition",
    "gold/dim_admission_source", "gold/dim_age_group",
]

for t in tables:
    n_parquet = spark.read.parquet(f"{base}/{t}").count()
    n_delta = spark.read.format("delta").load(f"{base}/{t}_delta").count()
    if n_parquet != n_delta:
        print("SKIPPED (counts differ):", t, n_parquet, n_delta)
        continue
    dbutils.fs.mv(f"{base}/{t}", True)
    dbutils.fs.mv(f"{base}/{t}_delta", f"{base}/{t}", True)
    print(t, "->", spark.read.format("delta").load(f"{base}/{t}").count(), "rows (Delta)")