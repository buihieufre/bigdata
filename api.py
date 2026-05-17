from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
import requests
import ta

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS = {}
THRESHOLDS = {}
TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']
BINANCE_INTERVALS = {'1min': '1m', '5min': '5m', '15min': '15m', '1H': '1h', '4H': '4h', '1D': '1d'}
FEATURES = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'return_1', 'return_5', 'volatility', 'volume']

def get_ema_periods(tf: str):
    if tf in ['1min', '5min', '15min']:
        return {"fast": 9, "slow": 21}
    elif tf in ['1H', '4H']:
        return {"fast": 20, "slow": 50}
    elif tf in ['1D']:
        return {"fast": 50, "slow": 100}
    return {"fast": 9, "slow": 21}

def load_models_and_data():
    global MODELS, THRESHOLDS
    
    # Load Thresholds
    thresholds_file = 'models/thresholds.json'
    if os.path.exists(thresholds_file):
        try:
            with open(thresholds_file, 'r') as f:
                THRESHOLDS = json.load(f)
        except Exception as e:
            logger.error(f"Error loading thresholds: {e}")
            
    # Load Models
    for tf in TIMEFRAMES:
        model_file = f'models/xgb_model_{tf}.pkl'
        if os.path.exists(model_file):
            try:
                MODELS[tf] = joblib.load(model_file)
            except Exception as e:
                logger.error(f"Error loading model {tf}: {e}")

load_models_and_data()

def fetch_missing_candles(symbol: str, interval: str, start_ts: int):
    """Fetch missing candles from Binance from start_ts (in ms) to now"""
    url = "https://api.binance.com/api/v3/klines"
    klines = []
    current_start = start_ts
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'startTime': current_start
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            klines.extend(data)
            current_start = data[-1][0] + 1
            
            # If we fetched less than 1000, we are at current time
            if len(data) < 1000:
                break
                
        except Exception as e:
            logger.error(f"Error fetching missing candles: {e}")
            break
            
    if not klines:
        return pd.DataFrame()
        
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.as_unit('us')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def calculate_features(df: pd.DataFrame, tf: str):
    periods = get_ema_periods(tf)
    
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=periods['fast'])
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=periods['slow'])
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['volatility'] = df['return_1'].rolling(window=20).std()
    
    # We do NOT generate targets or shift here because we are only predicting
    # Also we keep NaNs to maintain row count, but XGBoost handles NaNs or we can bfill
    return df

@app.route('/api/hdfs-status', methods=['GET'])
def hdfs_status():
    """Kiểm tra trạng thái dữ liệu trên HDFS"""
    import subprocess
    try:
        result = subprocess.run(
            ['hdfs', 'dfs', '-ls', '-h', '/user/hdoop/bigdata/'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return jsonify({"status": "connected", "content": result.stdout})
        else:
            return jsonify({"status": "error", "message": result.stderr})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def read_features_from_hdfs(timeframe: str) -> pd.DataFrame:
    """
    Đọc features: ưu tiên local file (nhanh hơn cho API server).
    HDFS chỉ dùng khi local không có.
    """
    local_file = f'data/btc_features_{timeframe}.parquet'
    
    # Ưu tiên đọc local (nhanh hơn cho API real-time)
    if os.path.exists(local_file):
        df = pd.read_parquet(local_file)
        logger.info(f"[{timeframe}] Đọc {len(df)} dòng từ local")
        return df
    
    # Fallback: đọc từ HDFS bằng subprocess (tránh lỗi CLASSPATH của PyArrow)
    try:
        import subprocess
        import tempfile
        hdfs_path = f'/user/hdoop/bigdata/features/btc_features_{timeframe}.parquet'
        tmp_file = f'/tmp/btc_features_{timeframe}_hdfs.parquet'
        result = subprocess.run(
            ['hdfs', 'dfs', '-get', '-f', hdfs_path, tmp_file],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            df = pd.read_parquet(tmp_file)
            logger.info(f"[{timeframe}] Đọc {len(df)} dòng từ HDFS (via CLI)")
            return df
    except Exception as e:
        logger.warning(f"[{timeframe}] HDFS fallback failed: {e}")
    
    raise FileNotFoundError(f"Data for {timeframe} not found locally or on HDFS")


@app.route('/api/data', methods=['GET'])
def get_data():
    timeframe = request.args.get('timeframe', '1min')
    if timeframe not in TIMEFRAMES:
        timeframe = '1min'
        
    try:
        df = read_features_from_hdfs(timeframe)
        
        # Keep only the last 1000 rows to ensure fast processing but enough history for EMAs
        df = df.tail(1000).copy()
        
        # Determine the timestamp of the last historical candle
        last_ts = int(df['timestamp'].max().timestamp() * 1000)
        
        # Fetch missing candles from Binance
        interval = BINANCE_INTERVALS[timeframe]
        # Request from last_ts + 1 ms to avoid duplicating the very last candle
        new_candles = fetch_missing_candles('BTCUSDT', interval, last_ts + 1)
        
        if not new_candles.empty:
            logger.info(f"Patched {len(new_candles)} missing candles for {timeframe}")
            
            # Extract raw OHLCV from the historical tail
            raw_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Build fresh DataFrames to avoid pandas internal block manager issues
            # and normalize datetime resolution (parquet uses us, Binance API uses ns)
            historical_raw = pd.DataFrame({
                'timestamp': df['timestamp'].values.astype('datetime64[us]'),
                'open': df['open'].values.astype(float),
                'high': df['high'].values.astype(float),
                'low': df['low'].values.astype(float),
                'close': df['close'].values.astype(float),
                'volume': df['volume'].values.astype(float),
            })
            new_candles_clean = pd.DataFrame({
                'timestamp': new_candles['timestamp'].values.astype('datetime64[us]'),
                'open': new_candles['open'].values.astype(float),
                'high': new_candles['high'].values.astype(float),
                'low': new_candles['low'].values.astype(float),
                'close': new_candles['close'].values.astype(float),
                'volume': new_candles['volume'].values.astype(float),
            })
            
            # Concatenate historical and new candles
            merged_raw = pd.concat([historical_raw, new_candles_clean], ignore_index=True)
            merged_raw.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            merged_raw = merged_raw.sort_values('timestamp').reset_index(drop=True)
            
            # Recalculate features on the merged data
            df = calculate_features(merged_raw, timeframe)
        
        # Predict
        tf_thresholds = THRESHOLDS.get(timeframe, {"buy": 0.65, "sell": 0.30})
        buy_thresh = tf_thresholds.get("buy", 0.65)
        sell_thresh = tf_thresholds.get("sell", 0.30)
        
        if timeframe in MODELS:
            # We predict on the entire tail. XGBoost handles NaN automatically, but let's ffill just in case
            features_df = df[FEATURES].ffill().bfill()
            probs = MODELS[timeframe].predict_proba(features_df)[:, 1]
            df['prob'] = probs
            df['signal'] = 'HOLD'
            df.loc[df['prob'] >= buy_thresh, 'signal'] = 'BUY'
            df.loc[df['prob'] <= sell_thresh, 'signal'] = 'SELL'
        else:
            df['prob'] = 0.5
            df['signal'] = 'HOLD'
            
        df['time'] = df['timestamp'].apply(lambda x: int(x.timestamp()))  # Unix timestamp (resolution-safe)
        df = df.replace({np.nan: None})
        
        result = df[['time', 'open', 'high', 'low', 'close', 'volume', 'signal', 'prob', 'ema_fast', 'ema_slow']].to_dict(orient='records')
        
        response_payload = {
            "data": result,
            "thresholds": {"buy": buy_thresh, "sell": sell_thresh},
            "ema_periods": get_ema_periods(timeframe)
        }
        
        return jsonify(response_payload)
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
