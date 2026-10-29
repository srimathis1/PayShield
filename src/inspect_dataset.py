from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("PayShield-Data-Inspection")
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

print("\n========== DATASET SCHEMA ==========")
df.printSchema()

print("\n========== FIRST 10 TRANSACTIONS ==========")
df.show(10, truncate=False)

print("\n========== TOTAL RECORD COUNT ==========")
print(f"Total records: {df.count():,}")

print("\n========== COLUMN COUNT ==========")
print(f"Total columns: {len(df.columns)}")
print(df.columns)

print("\n========== FRAUD DISTRIBUTION ==========")
df.groupBy("isFraud").count().show()

print("\n========== TRANSACTION TYPES ==========")
df.groupBy("type").count().orderBy("count", ascending=False).show()

spark.stop()
