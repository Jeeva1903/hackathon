from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Health-status boundaries
HEALTHY_SCORE_MIN = 80.0
WARNING_SCORE_MIN = 60.0
CRITICAL_SCORE_MAX = 40.0

@dp.materialized_view(
    name="ey_dbx_emea_hackathon_team1.dev_gold.gold_device_daily_summary",
    comment="Daily operational and health summary per device",
    cluster_by=["device_id", "event_date"]
)
def gold_device_daily_summary():

    df = spark.read.table(
        "ey_dbx_emea_hackathon_team1.dev_gold.gold_device_feature_timeseries"
    )

    return (
        df
        .groupBy(
            "device_id",
            "event_date"
        )
        .agg(
            F.count("*").alias("reading_count"),

            F.min("timestamp")
             .alias("first_reading_timestamp"),

            F.max("timestamp")
             .alias("last_reading_timestamp"),

            # Temperature
            F.round(
                F.avg("temperature_c"), 2
            ).alias("avg_temperature_c"),

            F.round(
                F.max("temperature_c"), 2
            ).alias("max_temperature_c"),

            F.round(
                F.avg("temperature_delta_c"), 2
            ).alias("avg_temperature_delta_c"),

            F.round(
                F.max("temperature_delta_c"), 2
            ).alias("max_temperature_delta_c"),

            # Electrical
            F.round(
                F.avg("voltage_v"), 2
            ).alias("avg_voltage_v"),

            F.round(
                F.avg("current_a"), 2
            ).alias("avg_current_a"),

            F.round(
                F.avg("power_w"), 2
            ).alias("avg_power_w"),

            F.round(
                F.max("power_w"), 2
            ).alias("max_power_w"),

            # Load
            F.round(
                F.avg("operational_load_pct"), 2
            ).alias("avg_operational_load_pct"),

            F.round(
                F.max("operational_load_pct"), 2
            ).alias("max_operational_load_pct"),

            # Performance
            F.round(
                F.avg("performance_score"), 2
            ).alias("avg_performance_score"),

            F.round(
                F.min("performance_score"), 2
            ).alias("min_performance_score"),

            F.round(
                F.avg("efficiency_pct"), 2
            ).alias("avg_efficiency_pct"),

            F.round(
                F.min("efficiency_pct"), 2
            ).alias("min_efficiency_pct"),

            # Reliability
            F.sum("error_delta")
             .alias("new_errors"),

            F.sum("warning_delta")
             .alias("new_warnings"),

            F.sum("runtime_delta_hours")
             .alias("runtime_hours_added"),

            F.sum("cycle_delta")
             .alias("cycles_added"),

            # Health
            F.round(
                F.avg("risk_score"), 2
            ).alias("avg_risk_score"),

            F.round(
                F.max("risk_score"), 2
            ).alias("max_risk_score"),

            F.round(
                F.avg("device_health_score"), 2
            ).alias("avg_device_health_score"),

            F.round(
                F.min("device_health_score"), 2
            ).alias("min_device_health_score"),

            # Conditions
            F.sum(
                F.col("high_temperature_flag").cast("int")
            ).alias("high_temperature_readings"),

            F.sum(
                F.col("high_load_flag").cast("int")
            ).alias("high_load_readings"),

            F.sum(
                F.col("low_performance_flag").cast("int")
            ).alias("low_performance_readings"),

            F.sum(
                F.col("low_efficiency_flag").cast("int")
            ).alias("low_efficiency_readings"),

            F.sum(
                F.when(
                    F.col("device_health_status") == "CRITICAL",
                    1
                ).otherwise(0)
            ).alias("critical_reading_count"),

            F.sum(
                F.when(
                    F.col("device_health_status") == "HIGH_RISK",
                    1
                ).otherwise(0)
            ).alias("high_risk_reading_count")
        )
        .withColumn(
            "daily_device_status",
            F.when(
                F.col("critical_reading_count") > 0,
                "CRITICAL"
            )
            .when(
                F.col("high_risk_reading_count") > 0,
                "HIGH_RISK"
            )
            .when(
                F.col("min_device_health_score")
                < HEALTHY_SCORE_MIN,
                "WARNING"
            )
            .otherwise("HEALTHY")
        )
        .withColumn(
            "daily_summary_generated_timestamp",
            F.current_timestamp()
        )
    )