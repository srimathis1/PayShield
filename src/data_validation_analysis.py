from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield-Validation-Analysis")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

file_path = "data/raw/paysim dataset.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(file_path)
)

total_records = df.count()

print("\n========== DATASET SUMMARY ==========")
print(f"Total records: {total_records:,}")

# 1. Exact duplicate analysis
print("\n========== EXACT DUPLICATE ANALYSIS ==========")

distinct_records = df.distinct().count()
exact_duplicate_count = total_records - distinct_records

print(f"Distinct rows: {distinct_records:,}")
print(f"Exact duplicate rows beyond first occurrences: "
      f"{exact_duplicate_count:,}")

# 2. Origin account balance comparison
print("\n========== ORIGIN BALANCE ANALYSIS ==========")

origin_analysis = df.withColumn(
    "origin_balance_difference",
    F.col("oldbalanceOrg") - F.col("newbalanceOrig")
).withColumn(
    "origin_difference_from_amount",
    F.abs(
        F.col("origin_balance_difference") - F.col("amount")
    )
)

origin_summary = origin_analysis.agg(
    F.sum(
        F.when(
            F.col("origin_difference_from_amount") <= 0.01,
            1
        ).otherwise(0)
    ).alias("matches_within_tolerance"),

    F.sum(
        F.when(
            F.col("origin_difference_from_amount") > 0.01,
            1
        ).otherwise(0)
    ).alias("does_not_match_within_tolerance")
)

origin_summary.show(truncate=False)

# 3. Destination account balance comparison
print("\n========== DESTINATION BALANCE ANALYSIS ==========")

destination_analysis = df.withColumn(
    "destination_balance_difference",
    F.col("newbalanceDest") - F.col("oldbalanceDest")
).withColumn(
    "destination_difference_from_amount",
    F.abs(
        F.col("destination_balance_difference") - F.col("amount")
    )
)

destination_summary = destination_analysis.agg(
    F.sum(
        F.when(
            F.col("destination_difference_from_amount") <= 0.01,
            1
        ).otherwise(0)
    ).alias("matches_within_tolerance"),

    F.sum(
        F.when(
            F.col("destination_difference_from_amount") > 0.01,
            1
        ).otherwise(0)
    ).alias("does_not_match_within_tolerance")
)

destination_summary.show(truncate=False)

# 4. Fraud statistics by transaction type
print("\n========== FRAUD RATE BY TRANSACTION TYPE ==========")

fraud_summary = df.groupBy("type").agg(
    F.count("*").alias("total_transactions"),
    F.sum("isFraud").alias("fraud_transactions"),
    F.round(
        F.avg("isFraud") * 100,
        4
    ).alias("fraud_percentage"),
    F.round(
        F.avg("amount"),
        2
    ).alias("average_amount")
).orderBy(
    F.col("fraud_percentage").desc()
)

fraud_summary.show(truncate=False)

spark.stop()
