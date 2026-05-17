"""
HDFS Manager - Module quản lý đọc/ghi dữ liệu trên HDFS
=========================================================
Module này cung cấp các hàm tiện ích để:
1. Upload dữ liệu từ local lên HDFS
2. Đọc dữ liệu từ HDFS bằng Spark
3. Ghi kết quả phân tích lên HDFS
4. Đồng bộ dữ liệu giữa local và HDFS
"""

import os
import subprocess
import logging
import pandas as pd
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Cấu hình HDFS
# ============================================================
HDFS_NAMENODE = "hdfs://127.0.0.1:9000"
HDFS_BASE_PATH = f"{HDFS_NAMENODE}/user/hdoop/bigdata"
HDFS_RAW_PATH = f"{HDFS_BASE_PATH}/raw"
HDFS_FEATURES_PATH = f"{HDFS_BASE_PATH}/features"
HDFS_MODELS_PATH = f"{HDFS_BASE_PATH}/models"
HDFS_OUTPUT_PATH = f"{HDFS_BASE_PATH}/output"

# Paths không có prefix (dùng cho hdfs CLI)
HDFS_CLI_BASE = "/user/hdoop/bigdata"
HDFS_CLI_RAW = f"{HDFS_CLI_BASE}/raw"
HDFS_CLI_FEATURES = f"{HDFS_CLI_BASE}/features"
HDFS_CLI_MODELS = f"{HDFS_CLI_BASE}/models"
HDFS_CLI_OUTPUT = f"{HDFS_CLI_BASE}/output"

LOCAL_DATA_DIR = "data"
LOCAL_MODELS_DIR = "models"


def get_spark_session(app_name: str = "CryptoHDFS") -> SparkSession:
    """
    Tạo SparkSession với cấu hình để đọc/ghi HDFS.
    Spark tự động nhận diện HDFS thông qua cấu hình Hadoop (core-site.xml).
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# 1. UPLOAD: Local → HDFS
# ============================================================
def upload_raw_data():
    """Upload dữ liệu thô (btc_raw.parquet) lên HDFS"""
    local_file = os.path.join(LOCAL_DATA_DIR, "btc_raw.parquet")
    hdfs_path = f"{HDFS_CLI_RAW}/btc_raw.parquet"
    
    if not os.path.exists(local_file):
        logger.error(f"File {local_file} không tồn tại. Chạy download_binance.py trước.")
        return False
    
    return _hdfs_put(local_file, hdfs_path)


def upload_features():
    """Upload tất cả file features lên HDFS"""
    success = True
    for tf in ['1min', '5min', '15min', '1H', '4H', '1D']:
        local_file = os.path.join(LOCAL_DATA_DIR, f"btc_features_{tf}.parquet")
        hdfs_path = f"{HDFS_CLI_FEATURES}/btc_features_{tf}.parquet"
        
        if os.path.exists(local_file):
            if not _hdfs_put(local_file, hdfs_path):
                success = False
        else:
            logger.warning(f"File {local_file} không tồn tại, bỏ qua.")
    return success


def upload_all():
    """Upload toàn bộ dữ liệu (raw + features + thresholds) lên HDFS"""
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU UPLOAD DỮ LIỆU LÊN HDFS")
    logger.info("=" * 60)
    
    # Tạo thư mục trên HDFS
    _hdfs_mkdir(HDFS_CLI_RAW)
    _hdfs_mkdir(HDFS_CLI_FEATURES)
    _hdfs_mkdir(HDFS_CLI_MODELS)
    _hdfs_mkdir(HDFS_CLI_OUTPUT)
    
    # Upload từng phần
    upload_raw_data()
    upload_features()
    
    # Upload thresholds
    thresholds_file = os.path.join(LOCAL_MODELS_DIR, "thresholds.json")
    if os.path.exists(thresholds_file):
        _hdfs_put(thresholds_file, f"{HDFS_CLI_MODELS}/thresholds.json")
    
    logger.info("=" * 60)
    logger.info("UPLOAD HOÀN TẤT")
    logger.info("=" * 60)
    
    # Hiển thị cấu trúc HDFS
    list_hdfs_structure()


# ============================================================
# 2. ĐỌC DỮ LIỆU TỪ HDFS BẰNG SPARK
# ============================================================
def read_raw_from_hdfs(spark: SparkSession) -> DataFrame:
    """
    Đọc dữ liệu thô từ HDFS bằng Spark.
    
    Spark đọc trực tiếp từ HDFS, dữ liệu được phân tán trên các DataNode.
    Mỗi block (128MB mặc định) được xử lý bởi 1 task riêng biệt.
    """
    hdfs_path = f"{HDFS_RAW_PATH}/btc_raw.parquet"
    logger.info(f"Đọc dữ liệu thô từ HDFS: {hdfs_path}")
    
    df = spark.read.parquet(hdfs_path)
    logger.info(f"Đã đọc {df.count()} dòng, {len(df.columns)} cột từ HDFS")
    return df


def read_features_from_hdfs(spark: SparkSession, timeframe: str) -> DataFrame:
    """
    Đọc dữ liệu features cho 1 timeframe cụ thể từ HDFS.
    
    Args:
        spark: SparkSession
        timeframe: '1min', '5min', '15min', '1H', '4H', '1D'
    """
    hdfs_path = f"{HDFS_FEATURES_PATH}/btc_features_{timeframe}.parquet"
    logger.info(f"Đọc features [{timeframe}] từ HDFS: {hdfs_path}")
    
    df = spark.read.parquet(hdfs_path)
    logger.info(f"[{timeframe}] Đã đọc {df.count()} dòng, {len(df.columns)} cột")
    return df


def read_all_features_from_hdfs(spark: SparkSession) -> dict:
    """
    Đọc features của tất cả timeframes từ HDFS.
    
    Returns:
        dict: {timeframe: DataFrame}
    """
    results = {}
    for tf in ['1min', '5min', '15min', '1H', '4H', '1D']:
        try:
            results[tf] = read_features_from_hdfs(spark, tf)
        except Exception as e:
            logger.error(f"Lỗi đọc features [{tf}]: {e}")
    return results


# ============================================================
# 3. GHI KẾT QUẢ LÊN HDFS BẰNG SPARK
# ============================================================
def write_features_to_hdfs(spark: SparkSession, pdf: pd.DataFrame, timeframe: str):
    """
    Ghi features (sau khi tính xong) lên HDFS bằng Spark.
    
    Chuyển đổi Pandas DataFrame → Spark DataFrame rồi ghi Parquet lên HDFS.
    Dùng mode 'overwrite' để cập nhật dữ liệu mới nhất.
    """
    hdfs_path = f"{HDFS_FEATURES_PATH}/btc_features_{timeframe}.parquet"
    logger.info(f"Ghi features [{timeframe}] lên HDFS: {hdfs_path}")
    
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("overwrite").parquet(hdfs_path)
    logger.info(f"[{timeframe}] Đã ghi {len(pdf)} dòng lên HDFS")


def write_signals_to_hdfs(spark: SparkSession, signals: list):
    """
    Ghi tín hiệu trading real-time lên HDFS.
    
    Dùng mode 'append' để thêm tín hiệu mới mà không ghi đè dữ liệu cũ.
    """
    if not signals:
        return
    
    hdfs_path = f"{HDFS_OUTPUT_PATH}/signals"
    logger.info(f"Ghi {len(signals)} tín hiệu lên HDFS: {hdfs_path}")
    
    pdf = pd.DataFrame(signals)
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("append").parquet(hdfs_path)
    logger.info("Đã ghi tín hiệu lên HDFS")


# ============================================================
# 4. TIỆN ÍCH HDFS
# ============================================================
def list_hdfs_structure():
    """Hiển thị cấu trúc thư mục trên HDFS"""
    logger.info("\n" + "=" * 60)
    logger.info("CẤU TRÚC DỮ LIỆU TRÊN HDFS")
    logger.info("=" * 60)
    
    result = subprocess.run(
        ["hdfs", "dfs", "-ls", "-R", "-h", HDFS_CLI_BASE],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        logger.error(f"Lỗi liệt kê HDFS: {result.stderr}")


def _hdfs_mkdir(path: str):
    """Tạo thư mục trên HDFS"""
    result = subprocess.run(
        ["hdfs", "dfs", "-mkdir", "-p", path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"Lỗi tạo thư mục HDFS {path}: {result.stderr}")


def _hdfs_put(local_path: str, hdfs_path: str) -> bool:
    """Upload 1 file từ local lên HDFS (ghi đè nếu đã tồn tại)"""
    logger.info(f"Upload: {local_path} → {hdfs_path}")
    
    result = subprocess.run(
        ["hdfs", "dfs", "-put", "-f", local_path, hdfs_path],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        # Lấy kích thước file trên HDFS
        size_result = subprocess.run(
            ["hdfs", "dfs", "-du", "-h", hdfs_path],
            capture_output=True, text=True
        )
        size_info = size_result.stdout.strip().split()[0] if size_result.returncode == 0 else "?"
        logger.info(f"  ✓ Upload thành công ({size_info})")
        return True
    else:
        logger.error(f"  ✗ Upload thất bại: {result.stderr}")
        return False


# ============================================================
# MAIN: Demo đầy đủ pipeline HDFS
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HDFS Manager - Quản lý dữ liệu trên HDFS")
    parser.add_argument("action", choices=["upload", "read", "list", "demo"],
                        help="upload: Upload dữ liệu lên HDFS | read: Đọc dữ liệu từ HDFS | list: Liệt kê HDFS | demo: Demo đầy đủ")
    args = parser.parse_args()
    
    if args.action == "upload":
        upload_all()
        
    elif args.action == "list":
        list_hdfs_structure()
        
    elif args.action == "read":
        spark = get_spark_session("HDFS_Read_Demo")
        
        logger.info("\n" + "=" * 60)
        logger.info("ĐỌC DỮ LIỆU TỪ HDFS BẰNG SPARK")
        logger.info("=" * 60)
        
        # Đọc dữ liệu thô
        df_raw = read_raw_from_hdfs(spark)
        logger.info("\n--- Schema dữ liệu thô ---")
        df_raw.printSchema()
        logger.info("--- 5 dòng đầu tiên ---")
        df_raw.show(5, truncate=False)
        
        # Đọc features
        for tf in ['1min', '5min', '15min', '1H', '4H', '1D']:
            df_feat = read_features_from_hdfs(spark, tf)
            df_feat.show(3, truncate=False)
        
        spark.stop()
        
    elif args.action == "demo":
        spark = get_spark_session("HDFS_Full_Demo")
        
        print("\n" + "=" * 60)
        print("DEMO ĐẦY ĐỦ: ĐỌC/GHI DỮ LIỆU TRÊN HỆ THỐNG PHÂN TÁN HDFS")
        print("=" * 60)
        
        # --- PHẦN 1: Đọc dữ liệu thô từ HDFS ---
        print("\n▶ PHẦN 1: Đọc dữ liệu thô (btc_raw.parquet) từ HDFS")
        print("-" * 50)
        df_raw = read_raw_from_hdfs(spark)
        df_raw.printSchema()
        df_raw.show(5, truncate=False)
        
        # Thống kê cơ bản trên Spark (phân tán)
        print("\n▶ PHẦN 2: Phân tích thống kê trên Spark (phân tán)")
        print("-" * 50)
        df_raw.describe().show()
        
        # --- PHẦN 3: Đọc features từ HDFS ---
        print("\n▶ PHẦN 3: Đọc features đã xử lý từ HDFS")
        print("-" * 50)
        features_dict = read_all_features_from_hdfs(spark)
        
        for tf, df_feat in features_dict.items():
            row_count = df_feat.count()
            col_count = len(df_feat.columns)
            print(f"  [{tf:>5}] {row_count:>10,} dòng × {col_count} cột")
        
        # --- PHẦN 4: Phân tích trên Spark từ HDFS ---
        print("\n▶ PHẦN 4: Phân tích dữ liệu trên Spark từ HDFS")
        print("-" * 50)
        
        from pyspark.sql.functions import avg, min as spark_min, max as spark_max, count, stddev
        
        if '1H' in features_dict:
            df_1h = features_dict['1H']
            print("  Thống kê features timeframe 1H:")
            df_1h.select(
                avg("close").alias("avg_close"),
                spark_min("close").alias("min_close"),
                spark_max("close").alias("max_close"),
                avg("rsi").alias("avg_rsi"),
                stddev("rsi").alias("std_rsi"),
                avg("volatility").alias("avg_volatility"),
                count("*").alias("total_rows")
            ).show(truncate=False)
        
        # --- PHẦN 5: Ghi kết quả phân tích lên HDFS ---
        print("\n▶ PHẦN 5: Ghi kết quả phân tích lên HDFS")
        print("-" * 50)
        
        if '1H' in features_dict:
            # Tạo summary statistics và ghi lên HDFS
            df_1h = features_dict['1H']
            summary = df_1h.describe()
            summary.write.mode("overwrite").parquet(f"{HDFS_OUTPUT_PATH}/analysis_summary")
            print("  ✓ Đã ghi analysis_summary lên HDFS")
            
            # Đọc lại kết quả từ HDFS để xác nhận
            print("\n  Đọc lại kết quả từ HDFS:")
            spark.read.parquet(f"{HDFS_OUTPUT_PATH}/analysis_summary").show(truncate=False)
        
        # Liệt kê cấu trúc HDFS cuối cùng
        print("\n▶ PHẦN 6: Cấu trúc dữ liệu cuối cùng trên HDFS")
        print("-" * 50)
        list_hdfs_structure()
        
        spark.stop()
        print("\n✓ Demo hoàn tất!")
