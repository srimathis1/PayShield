import os
import json
import time
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

start_time = time.time()

spark = (
    SparkSession.builder
    .appName("PayShield-Pipeline-Monitor")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

curated_path = "output/curated_transactions"
quarantine_path = "output/quarantine_transactions"
log_path = "logs/pipeline_runs.jsonl"

print("\n========== PAYSHIELD PIPELINE MONITOR ==========")

# 1. Read the processed dataset
df = spark.read.parquet(curated_path)

# 2. Calculate key metrics in one aggregation
metrics = df.agg(
    F.count("*").alias("total_records"),
    F.sum("isFraud").alias("fraud_labelled_records"),
    F.sum("isFlaggedFraud").alias("flagged_records"),
    F.sum("amount").alias("total_transaction_amount"),
    F.avg("amount").alias("average_transaction_amount")
).first().asDict()

# 3. Count records by transaction type
type_counts = {
    row["type"]: row["count"]
    for row in df.groupBy("type").count().collect()
}

# 4. Calculate output directory size
output_bytes = 0

for root, _, files in os.walk(curated_path):
    for filename in files:
        output_bytes += os.path.getsize(
            os.path.join(root, filename)
        )

# 5. Count quarantined records if the output exists
if os.path.exists(quarantine_path):
    quarantine_df = spark.read.parquet(quarantine_path)
    quarantine_count = quarantine_df.count()
else:
    quarantine_count = 0

# 6. Build the execution report
duration_seconds = round(time.time() - start_time, 2)

report = {
    "run_timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": "SUCCESS",
    "total_records": metrics["total_records"],
    "fraud_labelled_records": metrics["fraud_labelled_records"],
    "flagged_records": metrics["flagged_records"],
    "total_transaction_amount": metrics[
        "total_transaction_amount"
    ],
    "average_transaction_amount": metrics[
        "average_transaction_amount"
    ],
    "quarantined_records": quarantine_count,
    "transaction_type_counts": type_counts,
    "curated_output_size_mb": round(
        output_bytes / (1024 * 1024), 2
    ),
    "monitoring_duration_seconds": duration_seconds,
    "output_path": curated_path
}

# 7. Append one JSON record per run
os.makedirs("logs", exist_ok=True)

with open(log_path, "a", encoding="utf-8") as log_file:
    log_file.write(json.dumps(report) + "\n")

# 8. Display the report
print(json.dumps(report, indent=2, default=str))

print(f"\nMetrics saved to: {log_path}")

spark.stop()

