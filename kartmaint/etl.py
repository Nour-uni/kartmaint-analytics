"""ETL: raw PostgreSQL tables (or CSV exports) -> cleaned, partitioned Parquet.

Cleaning rules are explicit and counted, and every run writes data/clean/dq_report.json so you can
show exactly how many rows each rule touched.

Usage:
    python -m kartmaint.etl --source jdbc     # read from PostgreSQL (Docker setup)
    python -m kartmaint.etl --source csv      # read data/raw_csv/*.csv (no database needed)
"""
import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from kartmaint.config import CLEAN_DIR, EVENT_TYPES, JDBC_URL, PG, RAW_CSV_DIR, VALID_PARTS
from kartmaint.spark_utils import get_spark

SCHEMAS = {
    "karts": T.StructType(
        [
            T.StructField("kart_id", T.IntegerType()),
            T.StructField("model", T.StringType()),
            T.StructField("purchase_date", T.DateType()),
        ]
    ),
    "sessions": T.StructType(
        [
            T.StructField("session_id", T.LongType()),
            T.StructField("kart_id", T.IntegerType()),
            T.StructField("session_date", T.DateType()),
            T.StructField("hours_used", T.DoubleType()),
        ]
    ),
    "maintenance_events": T.StructType(
        [
            T.StructField("event_id", T.LongType()),
            T.StructField("kart_id", T.IntegerType()),
            T.StructField("event_date", T.DateType()),
            T.StructField("event_type", T.StringType()),
            T.StructField("part", T.StringType()),
            T.StructField("cost", T.DoubleType()),
            T.StructField("downtime_hours", T.DoubleType()),
            T.StructField("is_breakdown", T.IntegerType()),
        ]
    ),
}


def read_table(spark: SparkSession, name: str, source: str, raw_dir: Path = RAW_CSV_DIR) -> DataFrame:
    schema = SCHEMAS[name]
    if source == "jdbc":
        df = (
            spark.read.format("jdbc")
            .option("url", JDBC_URL)
            .option("dbtable", name)
            .option("user", PG["user"])
            .option("password", PG["password"])
            .option("driver", "org.postgresql.Driver")
            .load()
        )
        # normalise types (NUMERIC -> double, BOOLEAN -> int) so both sources look identical
        return df.select([F.col(f.name).cast(f.dataType).alias(f.name) for f in schema.fields])
    return (
        spark.read.option("header", True)
        .option("dateFormat", "yyyy-MM-dd")
        .schema(schema)
        .csv(str(raw_dir / f"{name}.csv"))
    )


def _median_by(df: DataFrame, value_col: str, keys: list) -> DataFrame:
    return df.where(F.col(value_col) > 0).groupBy(*keys).agg(
        F.percentile_approx(value_col, 0.5).alias(f"{value_col}_median")
    )


def clean_sessions(sessions: DataFrame, karts: DataFrame, cutoff, report: dict) -> DataFrame:
    n0 = sessions.count()
    s = sessions.dropDuplicates(["session_id"])
    report["sessions_duplicate_ids_removed"] = n0 - s.count()

    n1 = s.count()
    s = s.where(F.col("hours_used").between(0.05, 12.0))
    report["sessions_invalid_hours_removed"] = n1 - s.count()

    n2 = s.count()
    s = (
        s.join(karts.select("kart_id", "purchase_date"), "kart_id", "inner")
        .where((F.col("session_date") >= F.col("purchase_date")) & (F.col("session_date") <= F.lit(cutoff)))
        .drop("purchase_date")
    )
    report["sessions_unknown_kart_or_bad_date_removed"] = n2 - s.count()
    report["sessions_clean_rows"] = s.count()
    return s


def clean_events(events: DataFrame, karts: DataFrame, cutoff, report: dict) -> DataFrame:
    e = (
        events.withColumn("part", F.lower(F.trim("part")))
        .withColumn("event_type", F.lower(F.trim("event_type")))
        .withColumn("is_breakdown", F.col("is_breakdown").cast("boolean"))
    )
    n0 = e.count()

    e = e.where(F.col("part").isin(VALID_PARTS) & F.col("event_type").isin(EVENT_TYPES))
    report["events_unknown_part_or_type_removed"] = n0 - e.count()

    n1 = e.count()
    e = (
        e.join(karts.select("kart_id", "purchase_date"), "kart_id", "inner")
        .where((F.col("event_date") >= F.col("purchase_date")) & (F.col("event_date") <= F.lit(cutoff)))
        .drop("purchase_date")
    )
    report["events_bad_date_or_unknown_kart_removed"] = n1 - e.count()

    n2 = e.count()
    e = e.dropDuplicates(["kart_id", "event_date", "event_type", "part"])
    report["events_duplicates_removed"] = n2 - e.count()

    # impute missing/invalid numbers with the median of the same (event_type, part)
    report["events_null_cost_imputed"] = e.where(F.col("cost").isNull() | (F.col("cost") < 0)).count()
    report["events_negative_downtime_fixed"] = e.where(F.col("downtime_hours") < 0).count()
    e = e.withColumn("cost", F.when(F.col("cost") < 0, None).otherwise(F.col("cost")))
    e = e.withColumn("downtime_hours", F.when(F.col("downtime_hours") < 0, None).otherwise(F.col("downtime_hours")))
    keys = ["event_type", "part"]
    cost_med = _median_by(e, "cost", keys)
    down_med = _median_by(e, "downtime_hours", keys)
    e = (
        e.join(cost_med, keys, "left")
        .join(down_med, keys, "left")
        .withColumn("cost", F.round(F.coalesce("cost", "cost_median"), 2))
        .withColumn("downtime_hours", F.round(F.coalesce("downtime_hours", "downtime_hours_median"), 1))
        .drop("cost_median", "downtime_hours_median")
        .fillna({"cost": 0.0, "downtime_hours": 0.0})
    )
    report["events_clean_rows"] = e.count()
    return e


def run(source: str = "jdbc", raw_dir: Path = RAW_CSV_DIR, out_dir: Path = CLEAN_DIR) -> dict:
    spark = get_spark("kartmaint-etl")
    karts = read_table(spark, "karts", source, raw_dir).dropDuplicates(["kart_id"]).cache()
    sessions_raw = read_table(spark, "sessions", source, raw_dir)
    events_raw = read_table(spark, "maintenance_events", source, raw_dir)

    cutoff = sessions_raw.agg(F.max("session_date")).first()[0]   # data cut-off = last recorded session
    report = {"source": source, "cutoff_date": str(cutoff), "raw_sessions": sessions_raw.count(),
              "raw_events": events_raw.count()}

    sessions = clean_sessions(sessions_raw, karts, cutoff, report)
    events = clean_events(events_raw, karts, cutoff, report)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("sessions", sessions), ("events", events)):
        date_col = "session_date" if name == "sessions" else "event_date"
        (
            df.withColumn("year", F.year(date_col))
            .withColumn("month", F.month(date_col))
            .repartition("year", "month")
            .write.mode("overwrite")
            .partitionBy("year", "month")
            .parquet(str(out_dir / name))
        )
    karts.write.mode("overwrite").parquet(str(out_dir / "karts"))

    (out_dir / "dq_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    spark.stop()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["jdbc", "csv"], default="jdbc")
    ap.add_argument("--raw-dir", type=Path, default=RAW_CSV_DIR)
    ap.add_argument("--out-dir", type=Path, default=CLEAN_DIR)
    a = ap.parse_args()
    run(a.source, a.raw_dir, a.out_dir)
