import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =====================================================
# 1. CREATE SPARK SESSION
# =====================================================

spark = (
    SparkSession.builder
    .appName("PayShield-Transaction-Pipeline")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# =====================================================
# 2. READ RAW DATA
# =====================================================

input_path = os.environ.get(
    "PAYSHIELD_INPUT_PATH", "data/raw/paysim dataset.csv"
)

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_path)
)

print("\n========== RAW DATA ==========")
df.printSchema()

total_records = df.count()
print(f"Total input records: {total_records:,}")

# =====================================================
# 3. DEFINE DATA QUALITY RULES
# =====================================================

required_columns = df.columns

missing_condition = F.lit(False)

for column_name in required_columns:
    missing_condition = missing_condition | F.col(column_name).isNull()

# Also reject empty account identifiers and transaction types.
missing_condition = (
    missing_condition
    | (F.trim(F.col("nameOrig")) == "")
    | (F.trim(F.col("nameDest")) == "")
    | (F.trim(F.col("type")) == "")
)

invalid_condition = (
    missing_condition
    | (F.col("amount") < 0)
    | (F.col("oldbalanceOrg") < 0)
    | (F.col("newbalanceOrig") < 0)
    | (F.col("oldbalanceDest") < 0)
    | (F.col("newbalanceDest") < 0)
    | (~F.col("isFraud").isin(0, 1))
    | (~F.col("isFlaggedFraud").isin(0, 1))
    | (~F.col("type").isin(
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ))
    | (F.col("step") <= 0)
)

invalid_df = df.filter(invalid_condition)
valid_df = df.filter(~invalid_condition)

# =====================================================
# 4. COUNT VALID AND INVALID RECORDS
# =====================================================

invalid_count = invalid_df.count()
valid_count = total_records - invalid_count

print("\n========== VALIDATION RESULTS ==========")
print(f"Total records: {total_records:,}")
print(f"Valid records: {valid_count:,}")
print(f"Invalid records: {invalid_count:,}")

# =====================================================
# 5. SAVE INVALID RECORDS TO QUARANTINE
# =====================================================

quarantine_path = os.environ.get(
    "PAYSHIELD_QUARANTINE_PATH",
    "output/quarantine_transactions"
)

(
    invalid_df.write
    .mode("overwrite")
    .parquet(quarantine_path)
)

print(f"Quarantine data saved to: {quarantine_path}")

# =====================================================
# 6. FEATURE ENGINEERING
# =====================================================

features_df = (
    valid_df

    # Difference between origin balance before and after
    .withColumn(
        "origin_balance_delta",
        F.col("oldbalanceOrg") - F.col("newbalanceOrig")
    )

    # Difference between destination balance after and before
    .withColumn(
        "destination_balance_delta",
        F.col("newbalanceDest") - F.col("oldbalanceDest")
    )

    # Absolute difference between origin balance change and amount
    .withColumn(
        "origin_balance_error",
        F.abs(
            F.col("origin_balance_delta") - F.col("amount")
        )
    )

    # Absolute difference between destination balance change and amount
    .withColumn(
        "destination_balance_error",
        F.abs(
            F.col("destination_balance_delta") - F.col("amount")
        )
    )

    # Diagnostic indicators, not fraud predictions
    .withColumn(
        "origin_balance_matches",
        F.col("origin_balance_error") <= 0.01
    )

    .withColumn(
        "destination_balance_matches",
        F.col("destination_balance_error") <= 0.01
    )

    # Categorize transaction amounts
    .withColumn(
        "amount_category",
        F.when(F.col("amount") < 1000, "LOW")
         .when(F.col("amount") < 10000, "MEDIUM")
         .when(F.col("amount") < 100000, "HIGH")
         .otherwise("VERY_HIGH")
    )

    # Preserve the destination identifier and derive its first character
    .withColumn(
        "destination_account_prefix",
        F.substring(F.col("nameDest"), 1, 1)
    )
)

# =====================================================
# 7. ANALYTICS USING SPARK SQL
# =====================================================

features_df.createOrReplaceTempView("transactions")

print("\n========== FRAUD ANALYTICS ==========")

fraud_analytics = spark.sql("""
    SELECT
        type,
        COUNT(*) AS total_transactions,
        SUM(isFraud) AS fraud_transactions,
        ROUND(AVG(isFraud) * 100, 4) AS fraud_percentage,
        ROUND(AVG(amount), 2) AS average_amount,
        ROUND(SUM(amount), 2) AS total_transaction_amount
    FROM transactions
    GROUP BY type
    ORDER BY fraud_percentage DESC
""")

fraud_analytics.show(truncate=False)

print("\n========== AMOUNT CATEGORY ANALYTICS ==========")

amount_analytics = spark.sql("""
    SELECT
        amount_category,
        COUNT(*) AS total_transactions,
        SUM(isFraud) AS fraud_transactions,
        ROUND(AVG(isFraud) * 100, 4) AS fraud_percentage
    FROM transactions
    GROUP BY amount_category
    ORDER BY total_transactions DESC
""")

amount_analytics.show(truncate=False)

# =====================================================
# 8. SAVE CURATED DATA AS PARQUET
# =====================================================

output_path = os.environ.get(
    "PAYSHIELD_OUTPUT_PATH",
    "output/curated_transactions"
)

(
    features_df.write
    .mode("overwrite")
    .partitionBy("type")
    .parquet(output_path)
)

print("\n========== PIPELINE COMPLETED ==========")
print(f"Input records: {total_records:,}")
print(f"Valid records: {valid_count:,}")
print(f"Quarantined records: {invalid_count:,}")
print(f"Curated output: {output_path}")

spark.stop()
