"""
Batch Feature Engineering — ICT / SMC Edition
===============================================
Thay thế toàn bộ indicator-based features (EMA, MACD, RSI, Bollinger)
bằng ICT / Smart Money Concept features.

RSI và EMA được giữ lại CHỈ để benchmark / visualization,
không được đưa vào XGBoost training features.
"""

import pandas as pd
import numpy as np
import ta
import logging
import os

from ict_features import compute_all_ict_features, ICT_FEATURES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'hdfs://127.0.0.1:9000/user/hdoop/bigdata/raw/btc_raw.parquet'
TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']


def _ema_periods(tf: str):
    if tf in ['1min', '5min', '15min']:
        return 9, 21
    elif tf in ['1H', '4H']:
        return 20, 50
    return 50, 100


def process_timeframe(spark, df_raw: pd.DataFrame, tf: str):
    logger.info(f"Processing timeframe: {tf}...")

    if tf == '1min':
        df = df_raw.copy()
    else:
        df = df_raw.set_index('timestamp').resample(tf).agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum',
        }).dropna().reset_index()

    # ── ICT CORE FEATURES ─────────────────────────────────────────
    logger.info(f"  [{tf}] Computing ICT features...")
    df = compute_all_ict_features(df)

    # ── BENCHMARK ONLY (không dùng cho model) ─────────────────────
    fast, slow = _ema_periods(tf)
    df['rsi']      = ta.momentum.rsi(df['close'], window=14)
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=fast)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=slow)

    # ── TARGET ────────────────────────────────────────────────────
    # 1 = price up after 5 candles, 0 = down
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)

    # Drop rows without target and rows with too many NaN ICT features
    df = df.dropna(subset=ICT_FEATURES + ['target']).reset_index(drop=True)

    # Drop last 5 rows (target leakage)
    df = df.iloc[:-5]

    # ── Write directly to HDFS using Spark ───────────────────────
    from hdfs_manager import write_features_to_hdfs
    write_features_to_hdfs(spark, df, tf)
    logger.info(f"  [{tf}] ✓ Synced to HDFS")


def verify_hdfs_data(spark):
    """Đọc lại dữ liệu từ HDFS bằng Spark để xác nhận tính toàn vẹn"""
    try:
        from hdfs_manager import read_all_features_from_hdfs

        logger.info("\n" + "=" * 50)
        logger.info("XÁC NHẬN DỮ LIỆU TRÊN HDFS BẰNG SPARK")
        logger.info("=" * 50)

        features = read_all_features_from_hdfs(spark)

        for tf, df in features.items():
            row_count = df.count()
            col_count = len(df.columns)
            logger.info(f"  [{tf:>5}] HDFS: {row_count:>10,} rows × {col_count} cols ✓")

        logger.info("Xác nhận HDFS hoàn tất!\n")
    except Exception as e:
        logger.warning(f"Không thể xác nhận HDFS (HDFS có thể chưa chạy): {e}")


def main():
    from hdfs_manager import get_spark_session, read_raw_from_hdfs
    spark = get_spark_session("FeatureEngineering_ICT")

    logger.info(f"Loading raw data from {INPUT_FILE}...")
    try:
        df_spark = read_raw_from_hdfs(spark)
        df_raw = df_spark.toPandas()
    except Exception as e:
        logger.error(f"Cannot read {INPUT_FILE} from HDFS: {e}")
        spark.stop()
        return

    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"ICT Features: {ICT_FEATURES}")

    for tf in TIMEFRAMES:
        process_timeframe(spark, df_raw, tf)

    logger.info("ICT feature engineering complete for all timeframes.")
    verify_hdfs_data(spark)
    spark.stop()


if __name__ == "__main__":
    main()
