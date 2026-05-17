import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_thresholds():
    # Load test data (we'll just use the features file and do a simple split or just evaluate on the whole set for now to find optimal threshold)
    try:
        df = pd.read_csv('data/btc_features.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Split identical to train_model.py (80/20 time series split)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    features = ['rsi', 'ema9', 'ema21', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'return_1', 'return_5', 'volatility', 'volume']
    X_test = test_df[features]
    y_test = test_df['target']

    model = joblib.load('models/xgb_model.pkl')
    probs = model.predict_proba(X_test)[:, 1]

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
    evaluate_thresholds()
