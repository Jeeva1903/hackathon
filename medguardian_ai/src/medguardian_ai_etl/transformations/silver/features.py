from pyspark import pipelines as dp
from pyspark.sql.functions import *


silver_expectations ={
    "valid_device_id":"device_id is not null",
    "timestamp_not_null":
        "timestamp IS NOT NULL",

    "batch_id_not_null":
        "batch_id IS NOT NULL",

    # Percentage columns
    "valid_operational_load":
        "operational_load_pct BETWEEN 0 AND 100",

    "valid_performance_score":
        "performance_score BETWEEN 0 AND 100",

    "valid_efficiency":
        "efficiency_pct BETWEEN 0 AND 100",

    # Electrical measurements
    "valid_voltage":
        "voltage_v >= 0",

    "valid_current":
        "current_a >= 0",

    "valid_power":
        "power_w >= 0",

    # Counters cannot be negative
    "valid_error_count":
        "error_count >= 0",

    "valid_warning_count":
        "warning_count >= 0",

    "valid_runtime_hours":
        "runtime_hours >= 0",

    "valid_cycle_count":
        "cycle_count >= 0"
}


@dp.table(
    name='ey_dbx_emea_hackathon_team1.dev_silver.curated',
    comment='Silver: cleaned telemetry data for ML models'
)
@dp.expect_all_or_drop(silver_expectations)
def curated():
    return (
        spark.readStream.table('ey_dbx_emea_hackathon_team1.dev_bronze.med')
        .withColumn(
        "high_temperature_flag",
        col("temperature_c") > 80
        )

        .withColumn(
            "high_load_flag",
            col("operational_load_pct") > 90
        )
        .withColumn(
            "low_performance_flag",
            col("performance_score") < 60
        )
        .withColumn(
            "device_warning_flag",
            (col("warning_count") > 0) |
            (col("error_count") > 0)
        )
        .withColumn(
            'silver_processed_time',
            current_timestamp()
        )
        .fillna(
            {
                'error_count':0,
                'warning_count':0
            }
        )
        .withColumn(
            'temperature_delta_c',
            round((col('temperature_c')-col('ambient_temperature_c')), 2)
        )
        .withColumn(
            'device_id',
            (trim(col('device_id'))))
        .withColumn(
            'batch_id',
            trim(col('batch_id')))
        .withColumn(
            'sensor_status',
            upper(trim(col('device_id'))))
        .select(
            'device_id',
            'timestamp',
            'batch_id',
            'temperature_c',
            'ambient_temperature_c',
            'temperature_delta_c',
            'voltage_v',
            'current_a',
            'power_w',
            'operational_load_pct',
            'performance_score',
            'efficiency_pct',
            'error_count',
            'warning_count',
            'runtime_hours',
            'cycle_count',
            'sensor_status',
            'synthetic_flag',
            'device_warning_flag',
            'high_temperature_flag',
            'high_load_flag',
            'silver_processed_time'   
        )
    )