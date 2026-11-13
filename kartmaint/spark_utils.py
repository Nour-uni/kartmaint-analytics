import os
from pathlib import Path

from pyspark.sql import SparkSession

from kartmaint.config import PG_JAR


def get_spark(app_name: str = "kartmaint") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
    )
    if Path(PG_JAR).exists():  # JDBC driver baked into the Docker image
        builder = builder.config("spark.jars", PG_JAR)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
