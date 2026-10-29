from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Initialize Spark
spark = (
    SparkSession.builder
    .appName("PayShield-Data-Quality")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. Read the actual PaySim dataset
file_path = "data/raw/paysim dataset.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(file_path)
)

# 3. Define the columns to inspect
columns_to_check = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

# 4. Build missing-value checks
quality_checks = [
    F.sum(
        F.when(
            F.col(c).isNull() |
            (
                (F.col(c).cast("string") == "") &
                F.lit(True)
            ),
            1
        ).otherwise(0)
    ).alias(f"{c}_missing")
    for c in columns_to_check
]

# 5. Add validation checks
quality_checks.extend([
    F.sum(
        F.when(F.col("amount") < 0, 1).otherwise(0)
    ).alias("negative_amounts"),

    F.sum(
        F.when(
            (F.col("oldbalanceOrg") < 0) |
            (F.col("newbalanceOrig") < 0) |
            (F.col("oldbalanceDest") < 0) |
            (F.col("newbalanceDest") < 0),
            1
        ).otherwise(0)
    ).alias("negative_balance_records"),

    F.sum(
        F.when(~F.col("isFraud").isin(0, 1), 1).otherwise(0)
    ).alias("invalid_fraud_labels"),

    F.sum(
        F.when(
            ~F.col("isFlaggedFraud").isin(0, 1),
            1
        ).otherwise(0)
    ).alias("invalid_flag_labels"),

    F.sum(
        F.when(
            ~F.col("type").isin(
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "CASH_IN",
                "DEBIT"
            ),
            1
        ).otherwise(0)
    ).alias("invalid_transaction_types"),

    F.sum(
        F.when(F.col("step") <= 0, 1).otherwise(0)
    ).alias("invalid_time_steps"),

    F.sum(
        F.when(
            F.col("nameOrig").isNull() |
            F.col("nameDest").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_account_identifiers")
])

# 6. Execute the quality checks in one aggregation
report = df.agg(*quality_checks).first().asDict()

print("\n========== PAYSHIELD DATA QUALITY REPORT ==========")
print(f"Total records inspected: {df.count():,}")

for check_name, result in report.items():
    print(f"{check_name}: {result or 0}")

# 7. Calculate records failing basic validation rules
invalid_condition = (
    (F.col("amount") < 0) |
    (F.col("oldbalanceOrg") < 0) |
    (F.col("newbalanceOrig") < 0) |
    (F.col("oldbalanceDest") < 0) |
    (F.col("newbalanceDest") < 0) |
    (~F.col("isFraud").isin(0, 1)) |
    (~F.col("isFlaggedFraud").isin(0, 1)) |
    (~F.col("type").isin(
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    )) |
    (F.col("step") <= 0) |
    F.col("nameOrig").isNull() |
    F.col("nameDest").isNull()
)

invalid_count = df.filter(invalid_condition).count()

print("\n========== VALIDATION SUMMARY ==========")
print(f"Records failing validation rules: {invalid_count:,}")
print(f"Records passing these rules: {df.count() - invalid_count:,}")

spark.stop()
