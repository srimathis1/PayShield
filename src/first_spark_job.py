from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# 2. Create sample transaction data
transactions = [
    ("TXN001", "SUCCESS", 1000.0),
    ("TXN002", "FAILED", 2500.0),
    ("TXN003", "SUCCESS", 500.0),
    ("TXN004", "FAILED", 3000.0),
    ("TXN005", "SUCCESS", 1500.0)
]

columns = [
    "transaction_id",
    "status",
    "amount"
]

df = spark.createDataFrame(transactions, columns)

print("ORIGINAL TRANSACTIONS")
df.show()

# 3. Filter failed transactions
failed = df.filter(F.col("status") == "FAILED")

print("FAILED TRANSACTIONS")
failed.show()

# 4. Calculate transaction statistics
summary = df.groupBy("status").agg(
    F.count("*").alias("transaction_count"),
    F.sum("amount").alias("total_amount")
)

print("TRANSACTION SUMMARY")
summary.show()

# 5. Save the result in Parquet format
summary.write.mode("overwrite").parquet(
    "output/first_spark_job"
)

print("First Spark job completed!")

spark.stop()
