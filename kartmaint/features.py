"""Feature engineering (PySpark window functions).

Grain: one row per (kart, day). For a day D every feature uses ONLY information up to the end of
day D-1 (window frames end at -1), so there is no leakage from the day being predicted.

Label: breakdown_next_14d = 1 if the kart has a breakdown on any of days D+1 ... D+14.

Outputs:
    data/features/train_set   weekly (Monday) snapshots that have a complete 14-day future window
    data/features/latest      one row per kart for "tomorrow" (the day after the last data day), no label
"""
import argparse
from datetime import timedelta
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F

from kartmaint.config import BURN_IN_DAYS, CLEAN_DIR, FEATURES_DIR, HORIZON_DAYS, PARTS, SERVICE_INTERVAL_HOURS
from kartmaint.spark_utils import get_spark

NUMERIC_FEATURES = (
    ["age_days", "hours_7d", "hours_30d", "hours_90d", "sessions_30d"]
    + [f"hours_since_{p}" for p in PARTS]
    + ["max_wear_ratio"]
    + [
        "days_since_last_preventive",
        "days_since_last_breakdown",
        "breakdowns_90d",
        "breakdowns_365d",
        "downtime_90d",
        "cost_90d",
        "month",
    ]
)
CATEGORICAL_FEATURES = ["model"]


def build(clean_dir: Path = CLEAN_DIR, out_dir: Path = FEATURES_DIR):
    spark = get_spark("kartmaint-features")
    sessions = spark.read.parquet(str(clean_dir / "sessions"))
    events = spark.read.parquet(str(clean_dir / "events"))
    karts = spark.read.parquet(str(clean_dir / "karts"))

    min_day, max_day = sessions.agg(F.min("session_date"), F.max("session_date")).first()

    # 1) daily grid: every kart x every day since it was bought (+1 virtual day for scoring "tomorrow")
    calendar = spark.sql(
        f"SELECT explode(sequence(to_date('{min_day}'), to_date('{max_day + timedelta(days=1)}'), "
        f"interval 1 day)) AS day"
    )
    grid = karts.crossJoin(calendar).where(F.col("day") >= F.col("purchase_date"))

    daily_hours = sessions.groupBy("kart_id", F.col("session_date").alias("day")).agg(
        F.sum("hours_used").alias("hours"), F.count("*").alias("n_sessions")
    )
    daily_events = events.groupBy("kart_id", F.col("event_date").alias("day")).agg(
        F.sum(F.col("is_breakdown").cast("int")).alias("breakdown_cnt"),
        F.sum(F.when(F.col("event_type") == "preventive", 1).otherwise(0)).alias("preventive_cnt"),
        F.sum("downtime_hours").alias("downtime"),
        F.sum("cost").alias("cost"),
        *[
            F.sum(
                F.when((F.col("part") == p) & F.col("event_type").isin("preventive", "corrective"), 1).otherwise(0)
            ).alias(f"svc_{p}")
            for p in PARTS
        ],
    )
    df = (
        grid.join(daily_hours, ["kart_id", "day"], "left")
        .join(daily_events, ["kart_id", "day"], "left")
        .fillna(0)
    )

    # 2) window features. Frames end at -1 => "as of the end of the previous day".
    w = Window.partitionBy("kart_id").orderBy("day")
    w_prev = w.rowsBetween(Window.unboundedPreceding, -1)
    w_thru = w.rowsBetween(Window.unboundedPreceding, 0)

    def past(col, n):
        return F.sum(col).over(w.rowsBetween(-n, -1))

    df = df.withColumn("cum_thru", F.sum("hours").over(w_thru))
    cum_before = F.coalesce(F.sum("hours").over(w_prev), F.lit(0.0))
    for p in PARTS:   # hours of use since this part was last serviced (preventive or corrective)
        df = df.withColumn(f"svc_cum_{p}", F.when(F.col(f"svc_{p}") > 0, F.col("cum_thru")))
        last_cum = F.coalesce(F.last(f"svc_cum_{p}", ignorenulls=True).over(w_prev), F.lit(0.0))
        df = df.withColumn(f"hours_since_{p}", cum_before - last_cum)

    last_prev_date = F.last(F.when(F.col("preventive_cnt") > 0, F.col("day")), ignorenulls=True).over(w_prev)
    last_break_date = F.last(F.when(F.col("breakdown_cnt") > 0, F.col("day")), ignorenulls=True).over(w_prev)
    age = F.datediff("day", "purchase_date")

    df = (
        df.withColumn("age_days", age)
        .withColumn("hours_7d", F.coalesce(past("hours", 7), F.lit(0.0)))
        .withColumn("hours_30d", F.coalesce(past("hours", 30), F.lit(0.0)))
        .withColumn("hours_90d", F.coalesce(past("hours", 90), F.lit(0.0)))
        .withColumn("sessions_30d", F.coalesce(past("n_sessions", 30), F.lit(0)))
        .withColumn("days_since_last_preventive", F.coalesce(F.datediff("day", last_prev_date), age))
        .withColumn("days_since_last_breakdown", F.least(F.coalesce(F.datediff("day", last_break_date), F.lit(365)), F.lit(365)))
        .withColumn("breakdowns_90d", F.coalesce(past("breakdown_cnt", 90), F.lit(0)))
        .withColumn("breakdowns_365d", F.coalesce(past("breakdown_cnt", 365), F.lit(0)))
        .withColumn("downtime_90d", F.coalesce(past("downtime", 90), F.lit(0.0)))
        .withColumn("cost_90d", F.coalesce(past("cost", 90), F.lit(0.0)))
        .withColumn("month", F.month("day"))
        # how overdue is the MOST overdue part, relative to its recommended service interval
        .withColumn(
            "max_wear_ratio",
            F.greatest(*[F.col(f"hours_since_{p}") / F.lit(SERVICE_INTERVAL_HOURS[p]) for p in PARTS]),
        )
    )

    # 3) label: any breakdown in the next 14 days (only when the whole window lies inside the data)
    future = F.sum("breakdown_cnt").over(w.rowsBetween(1, HORIZON_DAYS))
    last_labelled_day = F.date_sub(F.lit(max_day), HORIZON_DAYS)
    df = df.withColumn(
        "breakdown_next_14d", F.when(F.col("day") <= last_labelled_day, (future > 0).cast("int"))
    )

    keep = ["kart_id", "day", "model"] + NUMERIC_FEATURES + ["breakdown_next_14d"]
    first_ok_day = F.date_add(F.lit(min_day), BURN_IN_DAYS)

    train = (
        df.where(
            (F.col("day") >= first_ok_day)
            & F.col("breakdown_next_14d").isNotNull()
            & (F.dayofweek("day") == 2)   # Mondays only: weekly snapshots, less autocorrelation
        )
        .select(*keep)
    )
    latest = df.where(F.col("day") == F.date_add(F.lit(max_day), 1)).select(*keep)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.coalesce(1).write.mode("overwrite").parquet(str(out_dir / "train_set"))
    latest.coalesce(1).write.mode("overwrite").parquet(str(out_dir / "latest"))

    n_train = train.count()
    pos = train.where(F.col("breakdown_next_14d") == 1).count()
    print(f"train rows: {n_train:,}  positives: {pos:,} ({pos / max(n_train, 1):.1%})")
    print(f"latest rows (karts to score): {latest.count()}  as of {max_day}")
    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    ap.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    a = ap.parse_args()
    build(a.clean_dir, a.out_dir)
