from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ============================================================
# PROTOTYPE THRESHOLDS
# ============================================================

HIGH_TEMPERATURE_C = 80.0
CRITICAL_TEMPERATURE_C = 90.0

HIGH_TEMPERATURE_DELTA_C = 45.0

HIGH_LOAD_PCT = 90.0

LOW_PERFORMANCE_SCORE = 60.0
CRITICAL_PERFORMANCE_SCORE = 40.0

LOW_EFFICIENCY_PCT = 65.0
CRITICAL_EFFICIENCY_PCT = 45.0


# Health-status boundaries
HEALTHY_SCORE_MIN = 80.0
WARNING_SCORE_MIN = 60.0
CRITICAL_SCORE_MAX = 40.0

gold_feature_expectations = {
    "device_id_present":
        "device_id IS NOT NULL",

    "timestamp_present":
        "timestamp IS NOT NULL",

    "valid_health_score":
        "device_health_score BETWEEN 0 AND 100",

    "valid_risk_score":
        "risk_score BETWEEN 0 AND 100",

    "valid_thermal_stress":
        "thermal_stress_score BETWEEN 0 AND 100",

    "valid_load_stress":
        "load_stress_score BETWEEN 0 AND 100",

    "valid_performance_stress":
        "performance_stress_score BETWEEN 0 AND 100",

    "valid_efficiency_stress":
        "efficiency_stress_score BETWEEN 0 AND 100",

    "valid_reliability_stress":
        "reliability_stress_score BETWEEN 0 AND 100"
}

@dp.expect_all_or_drop(gold_feature_expectations)
@dp.materialized_view(
    name="ey_dbx_emea_hackathon_team1.dev_gold.gold_device_feature_timeseries",
    comment=(
        "Engineered medical-device health, stress, degradation "
        "and reliability features"
    ),
    cluster_by=["device_id", "event_date"]
)
def gold():
    df = spark.read.table('ey_dbx_emea_hackathon_team1.dev_silver.curated').withColumn('event_date', F.to_date(F.col('timestamp')))

    ##### Window Specification #####
    device_window = Window.partitionBy('device_id').orderBy('timestamp')

    window_5 = device_window.rowsBetween(-4,0)
    window_10 = device_window.rowsBetween(-9,0)

    ### Getting the Row Previous Values using Lag Function ####

    df = (
        df
        .withColumn(
            "previous_timestamp",
            F.lag("timestamp").over(device_window)
        )
        .withColumn(
            "previous_temperature_c",
            F.lag("temperature_c").over(device_window)
        )
        .withColumn(
            "previous_load_pct",
            F.lag("operational_load_pct").over(device_window)
        )
        .withColumn(
            "previous_voltage_v",
            F.lag("voltage_v").over(device_window)
        )
        .withColumn(
            "previous_current_a",
            F.lag("current_a").over(device_window)
        )
        .withColumn(
            "previous_power_w",
            F.lag("power_w").over(device_window)
        )
        .withColumn(
            "previous_performance_score",
            F.lag("performance_score").over(device_window)
        )
        .withColumn(
            "previous_efficiency_pct",
            F.lag("efficiency_pct").over(device_window)
        )
        .withColumn(
            "previous_error_count",
            F.lag("error_count").over(device_window)
        )
        .withColumn(
            "previous_warning_count",
            F.lag("warning_count").over(device_window)
        )
        .withColumn(
            "previous_runtime_hours",
            F.lag("runtime_hours").over(device_window)
        )
        .withColumn(
            "previous_cycle_count",
            F.lag("cycle_count").over(device_window)
        )
    )

##### Change Features ####

    df = (
                df
                .withColumn(
                    "minutes_since_previous_reading",
                    (
                        F.unix_timestamp("timestamp")
                        - F.unix_timestamp("previous_timestamp")
                    ) / 60.0
                )
                .withColumn(
                    "temperature_change_c",
                    F.col("temperature_c")
                    - F.col("previous_temperature_c")
                )
                .withColumn(
                    "load_change_pct",
                    F.col("operational_load_pct")
                    - F.col("previous_load_pct")
                )
                .withColumn(
                    "voltage_change_v",
                    F.col("voltage_v")
                    - F.col("previous_voltage_v")
                )
                .withColumn(
                    "current_change_a",
                    F.col("current_a")
                    - F.col("previous_current_a")
                )
                .withColumn(
                    "power_change_w",
                    F.col("power_w")
                    - F.col("previous_power_w")
                )
                .withColumn(
                    "performance_change",
                    F.col("performance_score")
                    - F.col("previous_performance_score")
                )
                .withColumn(
                    "efficiency_change",
                    F.col("efficiency_pct")
                    - F.col("previous_efficiency_pct")
                )
            )

        # COUNTER RESET FLAGS

    df = (
                df
                .withColumn(
                    "error_counter_reset_flag",
                    F.coalesce(
                        F.col("error_count")
                        < F.col("previous_error_count"),
                        F.lit(False)
                    )
                )
                .withColumn(
                    "warning_counter_reset_flag",
                    F.coalesce(
                        F.col("warning_count")
                        < F.col("previous_warning_count"),
                        F.lit(False)
                    )
                )
                .withColumn(
                    "runtime_counter_reset_flag",
                    F.coalesce(
                        F.col("runtime_hours")
                        < F.col("previous_runtime_hours"),
                        F.lit(False)
                    )
                )
                .withColumn(
                    "cycle_counter_reset_flag",
                    F.coalesce(
                        F.col("cycle_count")
                        < F.col("previous_cycle_count"),
                        F.lit(False)
                    )
                )
            )

      # COUNTER DELTAS
    df = (
        df
        .withColumn(
            "error_delta",
            F.when(
                F.col("previous_error_count").isNull()
                | F.col("error_counter_reset_flag"),
                0
            ).otherwise(
                F.col("error_count")
                - F.col("previous_error_count")
            )
        )
        .withColumn(
            "warning_delta",
            F.when(
                F.col("previous_warning_count").isNull()
                | F.col("warning_counter_reset_flag"),
                0
            ).otherwise(
                F.col("warning_count")
                - F.col("previous_warning_count")
            )
        )
        .withColumn(
            "runtime_delta_hours",
            F.when(
                F.col("previous_runtime_hours").isNull()
                | F.col("runtime_counter_reset_flag"),
                0
            ).otherwise(
                F.col("runtime_hours")
                - F.col("previous_runtime_hours")
            )
        )
        .withColumn(
            "cycle_delta",
            F.when(
                F.col("previous_cycle_count").isNull()
                | F.col("cycle_counter_reset_flag"),
                0
            ).otherwise(
                F.col("cycle_count")
                - F.col("previous_cycle_count")
            )
        )
    )

     # TEMPERATURE FEATURES

    df = (
        df
        .withColumn(
            "avg_temperature_5",
            F.avg("temperature_c").over(window_5)
        )
        .withColumn(
            "avg_temperature_10",
            F.avg("temperature_c").over(window_10)
        )
        .withColumn(
            "min_temperature_10",
            F.min("temperature_c").over(window_10)
        )
        .withColumn(
            "max_temperature_10",
            F.max("temperature_c").over(window_10)
        )
        .withColumn(
            "temperature_stddev_10",
            F.coalesce(
                F.stddev_samp("temperature_c").over(window_10),
                F.lit(0.0)
            )
        )
        .withColumn(
            "avg_temperature_delta_10",
            F.avg("temperature_delta_c").over(window_10)
        )
        .withColumn(
            "max_temperature_delta_10",
            F.max("temperature_delta_c").over(window_10)
        )
        .withColumn(
            "temperature_vs_recent_avg",
            F.col("temperature_c")
            - F.col("avg_temperature_10")
        )
        .withColumn(
            "temperature_range_10",
            F.col("max_temperature_10")
            - F.col("min_temperature_10")
        )
    )

    ## Load Features

    df = (
        df
        .withColumn(
            "avg_load_5",
            F.avg("operational_load_pct").over(window_5)
        )
        .withColumn(
            "avg_load_10",
            F.avg("operational_load_pct").over(window_10)
        )
        .withColumn(
            "max_load_10",
            F.max("operational_load_pct").over(window_10)
        )
        .withColumn(
            "load_stddev_10",
            F.coalesce(
                F.stddev_samp(
                    "operational_load_pct"
                ).over(window_10),
                F.lit(0.0)
            )
        )
        .withColumn(
            "high_load_count_10",
            F.sum(
                F.when(
                    F.col("operational_load_pct")
                    >= HIGH_LOAD_PCT,
                    1
                ).otherwise(0)
            ).over(window_10)
        )
        .withColumn(
            "records_in_window_10",
            F.count("*").over(window_10)
        )
        .withColumn(
            "high_load_ratio_10",
            F.col("high_load_count_10")
            / F.col("records_in_window_10")
        )
    )

# ELECTRICAL FEATURES
    df = (
        df
        .withColumn(
            "avg_voltage_10",
            F.avg("voltage_v").over(window_10)
        )
        .withColumn(
            "avg_current_10",
            F.avg("current_a").over(window_10)
        )
        .withColumn(
            "avg_power_10",
            F.avg("power_w").over(window_10)
        )
        .withColumn(
            "voltage_stddev_10",
            F.coalesce(
                F.stddev_samp("voltage_v").over(window_10),
                F.lit(0.0)
            )
        )
        .withColumn(
            "current_stddev_10",
            F.coalesce(
                F.stddev_samp("current_a").over(window_10),
                F.lit(0.0)
            )
        )
        .withColumn(
            "power_stddev_10",
            F.coalesce(
                F.stddev_samp("power_w").over(window_10),
                F.lit(0.0)
            )
        )
        .withColumn(
            "voltage_deviation_from_avg",
            F.abs(
                F.col("voltage_v") - F.col("avg_voltage_10")
            )
        )
        .withColumn(
            "current_deviation_from_avg",
            F.abs(
                F.col("current_a") - F.col("avg_current_10")
            )
        )
        .withColumn(
            "power_deviation_from_avg",
            F.abs(
                F.col("power_w") - F.col("avg_power_10")
            )
        )
    )

    # PERFORMANCE AND EFFICIENCY FEATURES
    df = (
        df
        .withColumn(
            "avg_performance_5",
            F.avg("performance_score").over(window_5)
        )
        .withColumn(
            "avg_performance_10",
            F.avg("performance_score").over(window_10)
        )
        .withColumn(
            "avg_efficiency_5",
            F.avg("efficiency_pct").over(window_5)
        )
        .withColumn(
            "avg_efficiency_10",
            F.avg("efficiency_pct").over(window_10)
        )
        .withColumn(
            "performance_degradation_10",
            F.greatest(
                F.col("avg_performance_10")
                - F.col("performance_score"),
                F.lit(0.0)
            )
        )
        .withColumn(
            "efficiency_degradation_10",
            F.greatest(
                F.col("avg_efficiency_10")
                - F.col("efficiency_pct"),
                F.lit(0.0)
            )
        )
        .withColumn(
            "performance_efficiency_gap",
            F.abs(
                F.col("performance_score")
                - F.col("efficiency_pct")
            )
        )
    )
    # RELIABILITY FEATURES
    df = (
        df
        .withColumn(
            "new_errors_last_5",
            F.sum("error_delta").over(window_5)
        )
        .withColumn(
            "new_errors_last_10",
            F.sum("error_delta").over(window_10)
        )
        .withColumn(
            "new_warnings_last_5",
            F.sum("warning_delta").over(window_5)
        )
        .withColumn(
            "new_warnings_last_10",
            F.sum("warning_delta").over(window_10)
        )
        .withColumn(
            "runtime_added_last_10",
            F.sum("runtime_delta_hours").over(window_10)
        )
        .withColumn(
            "cycles_added_last_10",
            F.sum("cycle_delta").over(window_10)
        )
        .withColumn(
            "errors_per_runtime_hour_10",
            F.when(
                F.col("runtime_added_last_10") > 0,
                F.col("new_errors_last_10")
                / F.col("runtime_added_last_10")
            ).otherwise(0.0)
        )
        .withColumn(
            "warnings_per_runtime_hour_10",
            F.when(
                F.col("runtime_added_last_10") > 0,
                F.col("new_warnings_last_10")
                / F.col("runtime_added_last_10")
            ).otherwise(0.0)
        )
        .withColumn(
            "errors_per_100_cycles_10",
            F.when(
                F.col("cycles_added_last_10") > 0,
                (
                    F.col("new_errors_last_10")
                    / F.col("cycles_added_last_10")
                ) * 100
            ).otherwise(0.0)
        )
    )
    # STRESS SCORES: 0 LOW, 100 HIGH
    df = (
        df
        .withColumn(
            "thermal_stress_score",
            F.least(
                F.greatest(
                    (
                        F.col("temperature_delta_c")
                        / HIGH_TEMPERATURE_DELTA_C
                    ) * 100,
                    F.lit(0.0)
                ),
                F.lit(100.0)
            )
        )
        .withColumn(
            "load_stress_score",
            F.least(
                F.greatest(
                    (
                        F.col("operational_load_pct")
                        / HIGH_LOAD_PCT
                    ) * 100,
                    F.lit(0.0)
                ),
                F.lit(100.0)
            )
        )
        .withColumn(
            "performance_stress_score",
            F.least(
                F.greatest(
                    100 - F.col("performance_score"),
                    F.lit(0.0)
                ),
                F.lit(100.0)
            )
        )
        .withColumn(
            "efficiency_stress_score",
            F.least(
                F.greatest(
                    100 - F.col("efficiency_pct"),
                    F.lit(0.0)
                ),
                F.lit(100.0)
            )
        )
        .withColumn(
            "reliability_stress_score",
            F.least(
                (
                    F.col("new_errors_last_10") * 15
                    + F.col("new_warnings_last_10") * 5
                ).cast("double"),
                F.lit(100.0)
            )
        )
    )

     # COMBINED RISK AND HEALTH SCORE
    df = (
        df
        .withColumn(
            "risk_score",
            F.round(
                F.col("thermal_stress_score") * 0.25
                + F.col("load_stress_score") * 0.15
                + F.col("performance_stress_score") * 0.20
                + F.col("efficiency_stress_score") * 0.15
                + F.col("reliability_stress_score") * 0.25,
                2
            )
        )
        .withColumn(
            "device_health_score",
            F.round(
                F.greatest(
                    F.lit(0.0),
                    100 - F.col("risk_score")
                ),
                2
            )
        )
    )
    # CONDITION FLAGS
    df = (
        df
        .withColumn(
            "high_temperature_flag",
            F.col("temperature_c") >= HIGH_TEMPERATURE_C
        )
        .withColumn(
            "critical_temperature_flag",
            F.col("temperature_c")
            >= CRITICAL_TEMPERATURE_C
        )
        .withColumn(
            "high_temperature_delta_flag",
            F.col("temperature_delta_c")
            >= HIGH_TEMPERATURE_DELTA_C
        )
        .withColumn(
            "high_load_flag",
            F.col("operational_load_pct") >= HIGH_LOAD_PCT
        )
        .withColumn(
            "low_performance_flag",
            F.col("performance_score")
            <= LOW_PERFORMANCE_SCORE
        )
        .withColumn(
            "low_efficiency_flag",
            F.col("efficiency_pct")
            <= LOW_EFFICIENCY_PCT
        )
        .withColumn(
            "new_error_flag",
            F.col("error_delta") > 0
        )
        .withColumn(
            "new_warning_flag",
            F.col("warning_delta") > 0
        )
        .withColumn(
            "unstable_power_flag",
            F.when(
                F.col("avg_power_10") > 0,
                (
                    F.col("power_stddev_10")
                    / F.col("avg_power_10")
                ) > 0.10
            ).otherwise(False)
        )
        .withColumn(
            "counter_reset_detected",
            F.col("error_counter_reset_flag")
            | F.col("warning_counter_reset_flag")
            | F.col("runtime_counter_reset_flag")
            | F.col("cycle_counter_reset_flag")
        )
    )

    # ========================================================
    # HEALTH STATUS
    # ========================================================

    return (
        df
        .withColumn(
            "device_health_status",
            F.when(
                (F.col("device_health_score")
                 <= CRITICAL_SCORE_MAX)
                | F.col("critical_temperature_flag")
                | (
                    F.col("performance_score")
                    <= CRITICAL_PERFORMANCE_SCORE
                )
                | (
                    F.col("efficiency_pct")
                    <= CRITICAL_EFFICIENCY_PCT
                ),
                "CRITICAL"
            )
            .when(
                F.col("device_health_score")
                < WARNING_SCORE_MIN,
                "HIGH_RISK"
            )
            .when(
                F.col("device_health_score")
                < HEALTHY_SCORE_MIN,
                "WARNING"
            )
            .otherwise("HEALTHY")
        )
        .withColumn(
            "active_condition_count",
            F.col("high_temperature_flag").cast("int")
            + F.col("high_temperature_delta_flag").cast("int")
            + F.col("high_load_flag").cast("int")
            + F.col("low_performance_flag").cast("int")
            + F.col("low_efficiency_flag").cast("int")
            + F.col("new_error_flag").cast("int")
            + F.col("new_warning_flag").cast("int")
            + F.col("unstable_power_flag").cast("int")
        )
        .withColumn(
            "feature_generated_timestamp",
            F.current_timestamp()
        )
    )










