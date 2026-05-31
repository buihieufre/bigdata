from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (StructType, StructField, StringType,
                                DoubleType, BooleanType, LongType)
import os
import datetime

# ============================================================
# Buffer tín hiệu để ghi lên HDFS theo batch
# ============================================================
HDFS_OUTPUT_PATH = "hdfs://127.0.0.1:9000/user/hdoop/bigdata/output/signals"
SIGNAL_BUFFER = []
FLUSH_EVERY = 100  # Ghi lên HDFS mỗi 100 tín hiệu


def flush_signals_to_hdfs(spark):
    """Ghi buffer tín hiệu lên HDFS dưới dạng Parquet (append)"""
    global SIGNAL_BUFFER

    if not SIGNAL_BUFFER:
        return

    try:
        import pandas as pd
        pdf = pd.DataFrame(SIGNAL_BUFFER)
        sdf = spark.createDataFrame(pdf)
        sdf.write.mode("append").parquet(HDFS_OUTPUT_PATH)
        print(f"  ✓ Flushed {len(SIGNAL_BUFFER)} signals to HDFS: {HDFS_OUTPUT_PATH}")
        SIGNAL_BUFFER = []
    except Exception as e:
        print(f"  ✗ HDFS flush failed: {e}")


def process_batch(df, epoch_id):
    """
    Process each micro-batch.
    ICT features need full OHLCV + timestamp rolling window.
    """
    import pandas as pd
    from realtime_features import RealtimeICTFeatureGenerator
    from realtime_predict import RealtimeICTPredictor
    from pyspark.sql.functions import col

    global SIGNAL_BUFFER

    # ── Append raw data to HDFS Data Lake ──────────────────────
    try:
        raw_to_append = df.select(
            (col("timestamp") / 1000).cast("timestamp").alias("timestamp"),
            "open", "high", "low", "close", "volume"
        )
        raw_to_append.write.mode("append").parquet("hdfs://127.0.0.1:9000/user/hdoop/bigdata/raw/btc_raw.parquet")
    except Exception as e:
        print(f"  ✗ Failed to append raw data to HDFS: {e}")

    pdf = df.toPandas()

    if pdf.empty:
        return

    for _, row in pdf.iterrows():
        # Pass full OHLCV + timestamp_ms to ICT generator
        features = RealtimeICTFeatureGenerator.update(
            open_        = row['open'],
            high         = row['high'],
            low          = row['low'],
            close        = row['close'],
            volume       = row['volume'],
            timestamp_ms = int(row['timestamp']),
        )

        if features is not None:
            prediction = RealtimeICTPredictor.predict(features)

            output = {
                "symbol":           row['symbol'],
                "close":            row['close'],
                "bias":             prediction['bias'],
                "probability_up":   float(prediction['probability_up']),
                "probability_down": float(prediction['probability_down']),
                "ict_context":      prediction.get('ict_context', ''),
                "timestamp":        datetime.datetime.now().isoformat(),
            }

            print(f"ICT SIGNAL: {output}")

            SIGNAL_BUFFER.append(output)

            if len(SIGNAL_BUFFER) >= FLUSH_EVERY:
                flush_signals_to_hdfs(df.sparkSession)


def main():
    spark = SparkSession.builder \
        .appName("CryptoTradingAdvisor") \
        .config("spark.sql.streaming.checkpointLocation", "checkpoints/") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Schema mở rộng — full OHLCV + timestamp (ms) cho ICT features
    schema = StructType([
        StructField("symbol",    StringType(),  True),
        StructField("open",      DoubleType(),  True),
        StructField("high",      DoubleType(),  True),
        StructField("low",       DoubleType(),  True),
        StructField("close",     DoubleType(),  True),
        StructField("volume",    DoubleType(),  True),
        StructField("timestamp", LongType(),    True),   # kline open time ms UTC
        StructField("is_closed", BooleanType(), True),
    ])

    raw_stream = spark.readStream \
        .format("socket") \
        .option("host", "127.0.0.1") \
        .option("port", 9999) \
        .load()

    parsed_stream = raw_stream.select(
        from_json(col("value"), schema).alias("data")
    ).select("data.*")

    query = parsed_stream.writeStream \
        .foreachBatch(process_batch) \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()
