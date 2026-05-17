import pandas as pd
import ta
import logging
import os
import json
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'data/btc_raw.parquet'
TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']

def get_ema_periods(tf: str):
    if tf in ['1min', '5min', '15min']:
        return 9, 21
    elif tf in ['1H', '4H']:
        return 20, 50
    elif tf in ['1D']:
        return 50, 100
    return 9, 21

def process_timeframe(df_raw: pd.DataFrame, tf: str):
    logger.info(f"Processing timeframe: {tf}...")
    
    if tf == '1min':
        df = df_raw.copy()
    else:
        # Resample data
        df = df_raw.set_index('timestamp').resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
    fast_period, slow_period = get_ema_periods(tf)
    
    # Generate features
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=fast_period)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=slow_period)
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['volatility'] = df['return_1'].rolling(window=20).std()
    
    df = df.dropna().reset_index(drop=True)
    
    # Target
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Drop last 5 rows
    df = df.iloc[:-5]
    
    output_file = f'data/btc_features_{tf}.parquet'
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {tf} features to {output_file}. Shape: {df.shape}")
    
    # === Đồng bộ lên HDFS ===
    hdfs_path = f'/user/hdoop/bigdata/features/btc_features_{tf}.parquet'
    result = subprocess.run(
        ['hdfs', 'dfs', '-put', '-f', output_file, hdfs_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info(f"  ✓ Synced to HDFS: {hdfs_path}")
    else:
        logger.warning(f"  ✗ HDFS sync failed: {result.stderr.strip()}")

def verify_hdfs_data():
    """Đọc lại dữ liệu từ HDFS bằng Spark để xác nhận tính toàn vẹn"""
    try:
        from hdfs_manager import get_spark_session, read_all_features_from_hdfs
        
        logger.info("\n" + "=" * 50)
        logger.info("XÁC NHẬN DỮ LIỆU TRÊN HDFS BẰNG SPARK")
        logger.info("=" * 50)
        
        spark = get_spark_session("FeatureEngineering_Verify")
        features = read_all_features_from_hdfs(spark)
        
        for tf, df in features.items():
            row_count = df.count()
            col_count = len(df.columns)
            logger.info(f"  [{tf:>5}] HDFS: {row_count:>10,} dòng × {col_count} cột ✓")
        
        spark.stop()
        logger.info("Xác nhận HDFS hoàn tất!\n")
    except Exception as e:
        logger.warning(f"Không thể xác nhận HDFS (HDFS có thể chưa chạy): {e}")


def main():
    logger.info(f"Loading raw data from {INPUT_FILE}...")
    try:
        df_raw = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"{INPUT_FILE} not found. Please run download_binance.py first.")
        return
        
    for tf in TIMEFRAMES:
        process_timeframe(df_raw, tf)
        
    logger.info("Feature engineering complete for all timeframes.")
    
    # Xác nhận dữ liệu trên HDFS
    verify_hdfs_data()

if __name__ == "__main__":
    main()
