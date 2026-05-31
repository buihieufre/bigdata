"""
Model Training — ICT Feature Set
==================================
Train XGBoost trên 14 ICT/SMC features thay vì indicator-based features.
"""

import pandas as pd
import xgboost as xgb
import joblib
import logging
import os
import json
import numpy as np
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ict_features import ICT_FEATURES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']


def get_optimal_threshold(probs, y_true, target_class=1):
    best_thresh = 0.5
    best_score  = 0

    if target_class == 1:
        thresholds = np.arange(0.55, 0.85, 0.01)
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            if preds.sum() > 5:
                prec = precision_score(y_true, preds, zero_division=0)
                score = prec * (preds.sum() ** 0.1)
                if score > best_score and prec > 0.5:
                    best_score = score
                    best_thresh = thresh
    else:
        thresholds = np.arange(0.20, 0.45, 0.01)
        y_true_inv = 1 - y_true
        for thresh in thresholds:
            preds = (probs <= thresh).astype(int)
            if preds.sum() > 5:
                prec = precision_score(y_true_inv, preds, zero_division=0)
                score = prec * (preds.sum() ** 0.1)
                if score > best_score and prec > 0.5:
                    best_score = score
                    best_thresh = thresh

    return round(float(best_thresh), 2)


def train_for_timeframe(spark, tf: str, thresholds_dict: dict):
    logger.info(f"--- Training ICT Model for Timeframe: {tf} ---")
    model_file = f'models/xgb_ict_{tf}.pkl'

    # ── Load data directly from HDFS ────────────────────
    try:
        hdfs_path = f'hdfs://127.0.0.1:9000/user/hdoop/bigdata/features/btc_features_{tf}.parquet'
        sdf = spark.read.parquet(hdfs_path)
        df  = sdf.toPandas()
        logger.info(f"[{tf}] Read {len(df)} rows from HDFS")
    except Exception as e:
        logger.error(f"[{tf}] Failed to read HDFS data: {e}. Run feature_engineering.py first.")
        return

    # Verify ICT features present
    missing = [f for f in ICT_FEATURES if f not in df.columns]
    if missing:
        logger.error(f"[{tf}] Missing ICT features: {missing}. Re-run feature_engineering.py.")
        return

    # ── Train / Test Split (time-series 80/20) ────────────────────
    split_idx  = int(len(df) * 0.8)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]

    X_train = train_df[ICT_FEATURES].fillna(0)
    y_train = train_df['target']
    X_test  = test_df[ICT_FEATURES].fillna(0)
    y_test  = test_df['target']

    # ── XGBoost ───────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators        = 200,
        learning_rate       = 0.05,
        max_depth           = 6,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 3,
        random_state        = 42,
        eval_metric         = 'logloss',
        early_stopping_rounds = 15,
    )

    model.fit(
        X_train, y_train,
        eval_set  = [(X_test, y_test)],
        verbose   = False,
    )

    # ── Evaluate ──────────────────────────────────────────────────
    probs  = model.predict_proba(X_test)[:, 1]
    preds  = (probs >= 0.5).astype(int)
    acc    = accuracy_score(y_test, preds)
    prec   = precision_score(y_test, preds, zero_division=0)
    rec    = recall_score(y_test, preds, zero_division=0)
    f1     = f1_score(y_test, preds, zero_division=0)
    logger.info(f"[{tf}] Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # Feature importance (top 5)
    importances = pd.Series(model.feature_importances_, index=ICT_FEATURES)
    top5 = importances.nlargest(5)
    logger.info(f"[{tf}] Top ICT features:\n{top5.to_string()}")

    # ── Optimal Thresholds ────────────────────────────────────────
    buy_thresh  = get_optimal_threshold(probs, y_test, target_class=1)
    sell_thresh = get_optimal_threshold(probs, y_test, target_class=0)
    if buy_thresh  < 0.55: buy_thresh  = 0.65
    if sell_thresh > 0.45: sell_thresh = 0.35

    thresholds_dict[tf] = {"buy": buy_thresh, "sell": sell_thresh}
    logger.info(f"[{tf}] LONG thresh: {buy_thresh:.2f}  SHORT thresh: {sell_thresh:.2f}")

    # ── Save model locally ───────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_file)
    logger.info(f"[{tf}] Model saved locally → {model_file}")

    # ── Upload model lên HDFS ────────────────────────────────────
    hdfs_model_path = f"/user/hdoop/bigdata/models/xgb_ict_{tf}.pkl"
    result = subprocess.run(
        ['hdfs', 'dfs', '-put', '-f', model_file, hdfs_model_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info(f"[{tf}] ✓ Model uploaded to HDFS: {hdfs_model_path}\n")
    else:
        logger.warning(f"[{tf}] ✗ HDFS upload failed: {result.stderr.strip()}\n")


def sync_models_to_hdfs():
    """Upload toàn bộ model .pkl và thresholds.json từ local lên HDFS."""
    logger.info("=" * 60)
    logger.info("UPLOADING ALL MODELS TO HDFS")
    logger.info("=" * 60)

    # Tạo thư mục models trên HDFS
    subprocess.run(
        ['hdfs', 'dfs', '-mkdir', '-p', '/user/hdoop/bigdata/models'],
        capture_output=True, text=True
    )

    success_count = 0
    for tf in TIMEFRAMES:
        model_file = f'models/xgb_ict_{tf}.pkl'
        if not os.path.exists(model_file):
            logger.warning(f"[{tf}] Model file not found locally: {model_file}")
            continue
        hdfs_path = f'/user/hdoop/bigdata/models/xgb_ict_{tf}.pkl'
        result = subprocess.run(
            ['hdfs', 'dfs', '-put', '-f', model_file, hdfs_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"[{tf}] ✓ {model_file} → HDFS:{hdfs_path}")
            success_count += 1
        else:
            logger.warning(f"[{tf}] ✗ Upload failed: {result.stderr.strip()}")

    # Upload thresholds.json
    thresh_file = 'models/thresholds.json'
    if os.path.exists(thresh_file):
        result = subprocess.run(
            ['hdfs', 'dfs', '-put', '-f', thresh_file,
             '/user/hdoop/bigdata/models/thresholds.json'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("✓ thresholds.json → HDFS:/user/hdoop/bigdata/models/thresholds.json")
        else:
            logger.warning(f"✗ thresholds.json upload failed: {result.stderr.strip()}")

    logger.info(f"Upload complete: {success_count}/{len(TIMEFRAMES)} models synced to HDFS")


def main():
    thresholds_dict = {}
    from hdfs_manager import get_spark_session
    spark = get_spark_session("ICT_Train_All")

    for tf in TIMEFRAMES:
        train_for_timeframe(spark, tf, thresholds_dict)

    spark.stop()

    # Save thresholds locally
    os.makedirs('models', exist_ok=True)
    with open('models/thresholds.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=4)
    logger.info("Saved ICT thresholds → models/thresholds.json")

    # Sync thresholds + tất cả models lên HDFS
    sync_models_to_hdfs()


if __name__ == "__main__":
    main()
