from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
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
    Since technical indicators need a rolling window, we pass the data to realtime module.
    """
    import pandas as pd
    from realtime_features import RealtimeFeatureGenerator
    from realtime_predict import RealtimePredictor
    import json
    
    global SIGNAL_BUFFER
    
    # Collect to pandas for processing (in a real highly distributed setup we'd use mapInPandas
    # but for a single symbol stream, collecting is efficient enough)
    pdf = df.toPandas()
    
    if pdf.empty:
        return
        
    # Process each row
    for _, row in pdf.iterrows():
        # Add to window and get features
        features = RealtimeFeatureGenerator.update(row['close'], row['volume'])
        
        if features is not None:
            # Predict
            prediction = RealtimePredictor.predict(features)
            
            output = {
                "symbol": row['symbol'],
                "close": row['close'],
                "prediction": prediction['signal'],
                "probability": float(prediction['prob']),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            
            # Print to console
            print(f"REALTIME SIGNAL: {output}")
            
            # Save to local file (append)
            os.makedirs('output', exist_ok=True)
            with open('output/signals.jsonl', 'a') as f:
                f.write(json.dumps(output) + '\n')
            
            # Buffer cho HDFS
            SIGNAL_BUFFER.append(output)
            
            # Flush lên HDFS khi đủ batch
            if len(SIGNAL_BUFFER) >= FLUSH_EVERY:
                flush_signals_to_hdfs(df.sparkSession)


def main():
    spark = SparkSession.builder \
        .appName("CryptoTradingAdvisor") \
        .config("spark.sql.streaming.checkpointLocation", "checkpoints/") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    # Define schema for the incoming JSON
    schema = StructType([
        StructField("symbol", StringType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        StructField("is_closed", BooleanType(), True)
    ])
    
    # Read from TCP Socket
    raw_stream = spark.readStream \
        .format("socket") \
        .option("host", "127.0.0.1") \
        .option("port", 9999) \
        .load()
        
    # Parse JSON
    parsed_stream = raw_stream.select(
        from_json(col("value"), schema).alias("data")
    ).select("data.*")
    
    # We only want to process when the 1s candle is closed (optional, but good for stability)
    # However, to be fully realtime, we process everything.
    
    # Write stream
    query = parsed_stream.writeStream \
        .foreachBatch(process_batch) \
        .start()
        
    query.awaitTermination()

if __name__ == "__main__":
    main()
