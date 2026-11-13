# Interview notes: be able to explain every choice

1. **Why synthetic data?** Real client records are confidential and small. The generator lets me create
   millions of rows, inject realistic dirt, and know the ground truth, so I can TEST my ETL (it recovers the
   clean table exactly). I say clearly that the metrics reflect simulator assumptions.
2. **Why PySpark if the data fits in memory?** To show the pattern: JDBC ingestion, partitioned Parquet,
   window functions. The code scales by changing `--karts` and the Spark master, not by rewriting.
3. **How do you avoid data leakage?** Window frames end at -1, so day D only sees up to D-1. A pandas test
   recomputes features independently. The split is chronological with a gap equal to the label horizon.
4. **Why not accuracy?** ~9% positives: predicting "no breakdown" always gives ~91% accuracy and is useless.
   I use PR-AUC and precision in the top 10% (what a mechanic can actually inspect).
5. **Why did logistic regression win over gradient boosting?** Chosen on validation PR-AUC; with ~10k weekly
   rows and a mostly monotonic wear effect, a simple model is competitive. I report both.
6. **What is `max_wear_ratio`?** The most overdue part relative to its recommended interval: a domain
   feature that compresses four wear counters into one actionable number.
7. **What are the weaknesses?** Simulated data, left-censored history, slight under-prediction in the
   test period (seasonality), no drift monitoring or scheduling.
8. **What would you do with real data?** Validate against real breakdown logs, add sensor features
   (temperature, vibration) if available, recalibrate periodically, schedule with Airflow, monitor drift.
9. **Walk through the label.** `breakdown_next_14d` = any breakdown on days D+1..D+14. Rows without a full
   14-day future window are excluded from training and used only for scoring "tomorrow".
10. **How do you know the ETL works?** I inject known problems into a known clean table; a test asserts the
    ETL output equals the original clean row count and has no nulls/negatives/duplicates.
