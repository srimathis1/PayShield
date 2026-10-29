import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield-Performance-Tuning")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

path = "output/curated_transactions"

df = spark.read.parquet(path)

print("\n========== SPARK CONFIGURATION ==========")
print(
    "Shuffle partitions:",
    spark.conf.get("spark.sql.shuffle.partitions")
)
print(
    "Adaptive Query Execution:",
    spark.conf.get("spark.sql.adaptive.enabled")
)

# -------------------------------------------------
# TEST 1: PARTITION PRUNING
# -------------------------------------------------

print("\n========== PARTITION PRUNING TEST ==========")

transfer_df = df.filter(
    F.col("type") == "TRANSFER"
)

transfer_df.explain(mode="formatted")

start = time.perf_counter()

transfer_count = transfer_df.count()

transfer_seconds = time.perf_counter() - start

print(f"\nTRANSFER records: {transfer_count:,}")
print(f"Execution time: {transfer_seconds:.2f} seconds")

# -------------------------------------------------
# TEST 2: AGGREGATION WITH FEWER SHUFFLE PARTITIONS
# -------------------------------------------------

print("\n========== AGGREGATION TEST ==========")

summary = df.groupBy("type").agg(
    F.count("*").alias("transaction_count"),
    F.sum("amount").alias("total_amount"),
    F.sum("isFraud").alias("fraud_count")
)

summary.explain(mode="formatted")

start = time.perf_counter()

summary.show(truncate=False)

aggregation_seconds = time.perf_counter() - start

print(
    f"\nAggregation execution time: "
    f"{aggregation_seconds:.2f} seconds"
)

print("\n========== TUNING RUN COMPLETED ==========")

spark.stop()
