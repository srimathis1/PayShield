from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield-Pipeline-Tests")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

raw_path = "data/raw/paysim dataset.csv"
curated_path = "output/curated_transactions"

# Read the original dataset
raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(raw_path)
)

# Read the saved Parquet dataset
curated = spark.read.parquet(curated_path)

print("\n========== PAYSHIELD PIPELINE TESTS ==========")

# TEST 1: Record counts
raw_count = raw.count()
curated_count = curated.count()

print(f"\nRaw records: {raw_count:,}")
print(f"Curated records: {curated_count:,}")

assert raw_count == curated_count, (
    "FAIL: Raw and curated record counts differ"
)

print("PASS: Record counts match")

# TEST 2: Required columns
required_columns = {
    "step",
    "type",
    "amount",
    "isFraud",
    "isFlaggedFraud",
    "origin_balance_delta",
    "destination_balance_delta",
    "origin_balance_error",
    "destination_balance_error",
    "amount_category",
    "destination_account_prefix"
}

missing_columns = required_columns - set(curated.columns)

assert not missing_columns, (
    f"FAIL: Missing columns: {missing_columns}"
)

print("PASS: All required columns exist")

# TEST 3: Fraud label totals
raw_fraud_count = (
    raw.filter(F.col("isFraud") == 1).count()
)

curated_fraud_count = (
    curated.filter(F.col("isFraud") == 1).count()
)

print(f"\nRaw fraud-labelled records: {raw_fraud_count:,}")
print(f"Curated fraud-labelled records: {curated_fraud_count:,}")

assert raw_fraud_count == curated_fraud_count, (
    "FAIL: Fraud-labelled record counts differ"
)

print("PASS: Fraud labels preserved")

# TEST 4: Transaction counts by type
raw_types = (
    raw.groupBy("type")
    .count()
    .withColumnRenamed("count", "raw_count")
)

curated_types = (
    curated.groupBy("type")
    .count()
    .withColumnRenamed("count", "curated_count")
)

type_comparison = (
    raw_types.join(curated_types, on="type", how="full")
    .fillna(0, subset=["raw_count", "curated_count"])
    .withColumn(
        "counts_match",
        F.col("raw_count") == F.col("curated_count")
    )
)

print("\n========== TRANSACTION TYPE COMPARISON ==========")
type_comparison.orderBy("type").show(truncate=False)

mismatched_types = type_comparison.filter(
    ~F.col("counts_match")
).count()

assert mismatched_types == 0, (
    "FAIL: Transaction counts differ for one or more types"
)

print("PASS: Transaction-type counts preserved")

# TEST 5: Engineered feature null checks
feature_columns = [
    "origin_balance_delta",
    "destination_balance_delta",
    "origin_balance_error",
    "destination_balance_error",
    "amount_category",
    "destination_account_prefix"
]

null_conditions = [
    F.col(column_name).isNull()
    for column_name in feature_columns
]

null_feature_condition = null_conditions[0]

for condition in null_conditions[1:]:
    null_feature_condition = null_feature_condition | condition

null_feature_count = curated.filter(
    null_feature_condition
).count()

print(
    f"\nRecords with null engineered features: "
    f"{null_feature_count:,}"
)

assert null_feature_count == 0, (
    "FAIL: Some engineered features contain nulls"
)

print("PASS: Engineered features contain no nulls")

print("\n========== ALL PIPELINE TESTS PASSED ==========")

spark.stop()
