from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, DoubleType, BooleanType

spark = SparkSession.builder \
    .appName("CryptoStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# schema JSON
schema = StructType() \
    .add("symbol", StringType()) \
    .add("close", DoubleType()) \
    .add("volume", DoubleType()) \
    .add("is_closed", BooleanType())

# đọc từ socket
df = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# parse JSON
parsed = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# chỉ lấy nến đã đóng
filtered = parsed.filter(col("is_closed") == True)

# output
query = filtered.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()