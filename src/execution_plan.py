from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield-Execution-Plan")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

path = "output/curated_transactions"

df = spark.read.parquet(path)

print("\n========== QUERY 1: FILTER ==========")

# Select only high-value transactions
high_value = df.filter(F.col("amount") >= 100000)

high_value.select(
    "type",
    "amount",
    "isFraud"
).explain(mode="formatted")

print("\n========== QUERY 2: AGGREGATION ==========")

summary = (
    df.groupBy("type")
    .agg(
        F.count("*").alias("transaction_count"),
        F.sum("amount").alias("total_amount"),
        F.sum("isFraud").alias("fraud_count")
    )
)

summary.explain(mode="formatted")

print("\n========== EXECUTION RESULTS ==========")

high_value_count = high_value.count()

print(
    f"Transactions with amount >= 100000: "
    f"{high_value_count:,}"
)

summary.show(truncate=False)

spark.stop()

