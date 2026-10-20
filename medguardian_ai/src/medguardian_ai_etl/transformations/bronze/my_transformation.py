from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Sample CSV source (Databricks sample dataset)
SOURCE_DIR = "/Volumes/ey_dbx_emea_hackathon_team1/dev_landing/files_vol/"
@dp.table(
    name="med",
    comment="Bronze: raw customer CSV files ingested incrementally with Auto Loader"
)
def med():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("pathGlobFilter", "telemetry.csv")
        .load(SOURCE_DIR)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )