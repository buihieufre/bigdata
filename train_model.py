import pandas as pd
import xgboost as xgb
import joblib
import logging
import os
import json
import numpy as np
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']
FEATURES = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'return_1', 'return_5', 'volatility', 'volume']

def get_optimal_threshold(probs, y_true, target_class=1):
    best_thresh = 0.5
    best_score = 0
    
    if target_class == 1:
        thresholds = np.arange(0.55, 0.85, 0.01)
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            num_signals = preds.sum()
            if num_signals > 5:  # Minimum signals
                prec = precision_score(y_true, preds, zero_division=0)
                # We want to maximize precision, but also keep some frequency
                score = prec * (num_signals ** 0.1) 
                if score > best_score and prec > 0.5:
                    best_score = score
                    best_thresh = thresh
    else:
        thresholds = np.arange(0.20, 0.45, 0.01)
        y_true_inv = 1 - y_true
        for thresh in thresholds:
            preds = (probs <= thresh).astype(int)
            num_signals = preds.sum()
            if num_signals > 5:
                prec = precision_score(y_true_inv, preds, zero_division=0)
                score = prec * (num_signals ** 0.1)
                if score > best_score and prec > 0.5:
                    best_score = score
                    best_thresh = thresh
                    
    return round(float(best_thresh), 2)

def train_for_timeframe(tf: str, thresholds_dict: dict):
    logger.info(f"--- Training Model for Timeframe: {tf} ---")
    input_file = f'data/btc_features_{tf}.parquet'
    model_file = f'models/xgb_model_{tf}.pkl'
    
    try:
        # === Ưu tiên đọc từ HDFS ===
        hdfs_path = f'/user/hdoop/bigdata/features/btc_features_{tf}.parquet'
        try:
            from hdfs_manager import get_spark_session
            spark = get_spark_session(f"Train_{tf}")
            sdf = spark.read.parquet(hdfs_path)
            df = sdf.toPandas()
            spark.stop()
            logger.info(f"[{tf}] Đọc {len(df)} dòng từ HDFS: {hdfs_path}")
        except Exception as hdfs_err:
            logger.warning(f"[{tf}] Không đọc được từ HDFS ({hdfs_err}), dùng file local")
            df = pd.read_parquet(input_file)
    except FileNotFoundError:
        logger.error(f"File {input_file} not found. Skipping {tf}.")
        return

    # Train/Test Split (Time Series: 80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[FEATURES], train_df['target']
    X_test, y_test = test_df[FEATURES], test_df['target']
    
    # Initialize model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate & Optimize
    probs = model.predict_proba(X_test)[:, 1]
    
    buy_thresh = get_optimal_threshold(probs, y_test, target_class=1)
    sell_thresh = get_optimal_threshold(probs, y_test, target_class=0)
    
    # Fallbacks if optimization fails
    if buy_thresh < 0.55: buy_thresh = 0.65
    if sell_thresh > 0.45 or sell_thresh == 0.5: sell_thresh = 0.30
    
    thresholds_dict[tf] = {
        "buy": buy_thresh,
        "sell": sell_thresh
    }
    
    logger.info(f"[{tf}] Optimal BUY Threshold: {buy_thresh:.2f}")
    logger.info(f"[{tf}] Optimal SELL Threshold: {sell_thresh:.2f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_file)
    logger.info(f"[{tf}] Model saved to {model_file}\n")

def main():
    thresholds_dict = {}
    for tf in TIMEFRAMES:
        train_for_timeframe(tf, thresholds_dict)
        
    # Save thresholds locally
    os.makedirs('models', exist_ok=True)
    with open('models/thresholds.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=4)
    logger.info("Saved all optimal thresholds to models/thresholds.json")
    
    # === Đồng bộ thresholds lên HDFS ===
    result = subprocess.run(
        ['hdfs', 'dfs', '-put', '-f', 'models/thresholds.json', '/user/hdoop/bigdata/models/thresholds.json'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("✓ Synced thresholds.json to HDFS")
    else:
        logger.warning(f"✗ HDFS sync failed: {result.stderr.strip()}")

if __name__ == "__main__":
    main()
