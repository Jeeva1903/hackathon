from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

@dp.materialized_view(
    name="ey_dbx_emea_hackathon_team1.dev_gold.gold_device_alert_candidates",
    comment="Latest unhealthy devices requiring investigation",
    cluster_by=["alert_severity", "device_id"]
)
def gold_device_alert_candidates():

    df = spark.read.table(
        "ey_dbx_emea_hackathon_team1.dev_gold.gold_device_latest_health"
    )

    return (
        df
        .filter(
            F.col("device_health_status").isin(
                "WARNING",
                "HIGH_RISK",
                "CRITICAL"
            )
        )
        .withColumn(
            "alert_severity",
            F.when(
                F.col("device_health_status") == "CRITICAL",
                "P1"
            )
            .when(
                F.col("device_health_status") == "HIGH_RISK",
                "P2"
            )
            .otherwise("P3")
        )
        .withColumn(
            "alert_reason",
            F.concat_ws(
                "; ",

                F.when(
                    F.col("critical_temperature_flag"),
                    "Critical device temperature"
                ),

                F.when(
                    F.col("high_temperature_delta_flag"),
                    "High device-to-ambient temperature difference"
                ),

                F.when(
                    F.col("high_load_flag"),
                    "High operational load"
                ),

                F.when(
                    F.col("low_performance_flag"),
                    "Low performance score"
                ),

                F.when(
                    F.col("low_efficiency_flag"),
                    "Low efficiency"
                ),

                F.when(
                    F.col("new_error_flag"),
                    "New device errors detected"
                ),

                F.when(
                    F.col("new_warning_flag"),
                    "New device warnings detected"
                ),

                F.when(
                    F.col("unstable_power_flag"),
                    "Unstable power measurements"
                ),

                F.when(
                    F.col("counter_reset_detected"),
                    "Device counter reset detected"
                )
            )
        )
        .withColumn(
            "recommended_action_category",
            F.when(
                F.col("alert_severity") == "P1",
                "IMMEDIATE_INSPECTION"
            )
            .when(
                F.col("alert_severity") == "P2",
                "PRIORITY_DIAGNOSTIC"
            )
            .otherwise("MONITOR_AND_REVIEW")
        )
        .withColumn(
            "alert_generated_timestamp",
            F.current_timestamp()
        )
        .select(
            "device_id",
            "timestamp",
            "batch_id",

            "alert_severity",
            "device_health_status",
            "device_health_score",
            "risk_score",

            "alert_reason",
            "recommended_action_category",

            "temperature_c",
            "ambient_temperature_c",
            "temperature_delta_c",

            "operational_load_pct",
            "performance_score",
            "efficiency_pct",

            "error_delta",
            "warning_delta",
            "new_errors_last_10",
            "new_warnings_last_10",

            "thermal_stress_score",
            "load_stress_score",
            "performance_stress_score",
            "efficiency_stress_score",
            "reliability_stress_score",

            "active_condition_count",
            "sensor_status",
            "synthetic_flag",

            "alert_generated_timestamp"
        )
    )