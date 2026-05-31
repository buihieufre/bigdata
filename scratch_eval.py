import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ict_features import ICT_FEATURES

def evaluate_thresholds(tf='1min'):
    input_file = f'data/btc_features_{tf}.parquet'
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error loading data: {e}. Make sure to run feature_engineering.py first.")
        return

    # Split identical to train_model.py (80/20 time series split)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[ICT_FEATURES]
    y_test = test_df['target']

    model_file = f'models/xgb_ict_{tf}.pkl'
    try:
        model = joblib.load(model_file)
    except Exception as e:
        print(f"Error loading model: {e}. Run train_model.py first.")
        return
        
    probs = model.predict_proba(X_test.fillna(0))[:, 1]

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Base Rate (Up candles): {y_test.mean():.4f}\n")
    print(f"{'Threshold':<10} | {'Signals':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Win Rate':<10}")
    print("-" * 75)

    for thresh in thresholds:
        # BUY condition
        preds = (probs > thresh).astype(int)
        num_signals = preds.sum()
        
        if num_signals > 0:
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            # Win rate is exactly precision for the BUY class
            win_rate = precision
        else:
            precision = recall = f1 = win_rate = 0.0

        print(f"{thresh:<10.2f} | {num_signals:<10} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {win_rate:<10.4f}")

    print("\nSELL THRESHOLDS (Predicting 0)")
    print(f"{'Threshold':<10} | {'Signals':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Win Rate':<10}")
    print("-" * 75)
    
    sell_thresholds = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    # target == 0 means price goes down
    y_test_sell = 1 - y_test 
    
    for thresh in sell_thresholds:
        # SELL condition
        preds = (probs < thresh).astype(int)
        num_signals = preds.sum()
        
        if num_signals > 0:
            precision = precision_score(y_test_sell, preds, zero_division=0)
            recall = recall_score(y_test_sell, preds, zero_division=0)
            f1 = f1_score(y_test_sell, preds, zero_division=0)
            win_rate = precision
        else:
            precision = recall = f1 = win_rate = 0.0

        print(f"{thresh:<10.2f} | {num_signals:<10} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {win_rate:<10.4f}")

if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else '1min'
    print(f"Evaluating thresholds for timeframe: {tf}")
    evaluate_thresholds(tf)
