from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

@dp.expect(
    "valid_latest_health_score",
    "device_health_score BETWEEN 0 AND 100"
)
@dp.materialized_view(
    name="ey_dbx_emea_hackathon_team1.dev_gold.gold_device_latest_health",
    comment="Latest health and risk state for every device",
    cluster_by=["device_id"]
)
def gold_device_latest_health():

    df = spark.read.table(
        "ey_dbx_emea_hackathon_team1.dev_gold.gold_device_feature_timeseries"
    )

    latest_window = (
        Window
        .partitionBy("device_id")
        .orderBy(
            F.col("timestamp").desc(),
            F.col("feature_generated_timestamp").desc()
        )
    )

    return (
        df
        .withColumn(
            "latest_record_rank",
            F.row_number().over(latest_window)
        )
        .filter(
            F.col("latest_record_rank") == 1
        )
        .drop("latest_record_rank")
    )