import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield-Storage-Benchmark")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

csv_path = "data/raw/paysim dataset.csv"
parquet_path = "output/curated_transactions"


def directory_size_mb(path):
    total_bytes = 0

    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)

    for root, _, files in os.walk(path):
        for filename in files:
            full_path = os.path.join(root, filename)

            if os.path.isfile(full_path):
                total_bytes += os.path.getsize(full_path)

    return total_bytes / (1024 * 1024)


def benchmark_read(label, reader):
    start = time.perf_counter()

    df = reader()

    # Count forces Spark to execute the read.
    row_count = df.count()

    elapsed = time.perf_counter() - start

    print(f"\n{label}")
    print(f"Rows: {row_count:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    return row_count, elapsed


print("\n========== PAYSHIELD STORAGE BENCHMARK ==========")

csv_count, csv_seconds = benchmark_read(
    "CSV READ",
    lambda: (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
    )
)

parquet_count, parquet_seconds = benchmark_read(
    "PARQUET READ",
    lambda: spark.read.parquet(parquet_path)
)

csv_size = directory_size_mb(csv_path)
parquet_size = directory_size_mb(parquet_path)

print("\n========== STORAGE COMPARISON ==========")
print(f"CSV size: {csv_size:.2f} MB")
print(f"Parquet size: {parquet_size:.2f} MB")

if csv_size > 0:
    reduction = (1 - parquet_size / csv_size) * 100
    print(f"Storage size reduction: {reduction:.2f}%")

print("\n========== CORRECTNESS CHECK ==========")
print(f"CSV row count: {csv_count:,}")
print(f"Parquet row count: {parquet_count:,}")

if csv_count == parquet_count:
    print("PASS: Both datasets contain the same number of rows.")
else:
    print("FAIL: Row counts differ.")

print(
    "\nNote: These are single-run timings. "
    "They are not a controlled performance benchmark."
)

spark.stop()

