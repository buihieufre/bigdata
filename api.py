"""
Flask API — ICT / SMC Edition
==============================
Serves ICT-based signals với:
  - bias (LONG/SHORT/NEUTRAL)
  - probability_up / probability_down
  - confidence_score
  - ict_context (human-readable confluence)
  - market_context (active zones, killzone, htf_bias)

Benchmark columns (rsi, ema_fast, ema_slow) vẫn được trả về cho visualization.
"""

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

from ict_features import compute_all_ict_features, ICT_FEATURES, compute_session_score, compute_macro_window

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS     = {}
THRESHOLDS = {}
TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']
BINANCE_INTERVALS = {
    '1min': '1m', '5min': '5m', '15min': '15m',
    '1H':   '1h', '4H':   '4h', '1D':    '1d',
}

MACRO_LABEL = {0: "Outside Macro", 1: "London Macro", 2: "NY Macro ⚡"}
HTF_LABEL   = {1: "BULLISH", 0: "RANGING", -1: "BEARISH"}


def _ema_periods(tf: str):
    if tf in ['1min', '5min', '15min']:
        return {"fast": 9, "slow": 21}
    elif tf in ['1H', '4H']:
        return {"fast": 20, "slow": 50}
    return {"fast": 50, "slow": 100}


def load_models_and_thresholds():
    global MODELS, THRESHOLDS
    thresh_file = 'models/thresholds.json'
    if os.path.exists(thresh_file):
        with open(thresh_file) as f:
            THRESHOLDS = json.load(f)

    for tf in TIMEFRAMES:
        for model_path in [f'models/xgb_ict_{tf}.pkl', f'models/xgb_model_{tf}.pkl']:
            if os.path.exists(model_path):
                try:
                    MODELS[tf] = joblib.load(model_path)
                    logger.info(f"Loaded: {model_path}")
                except Exception as e:
                    logger.error(f"Error loading {model_path}: {e}")
                break


load_models_and_thresholds()


# ── Binance candle fetch ────────────────────────────────────────────────────
def fetch_missing_candles(symbol: str, interval: str, start_ts: int) -> pd.DataFrame:
    """Fetch missing candles from Binance REST API."""
    url    = "https://api.binance.com/api/v3/klines"
    klines = []
    current_start = start_ts

    while True:
        params = {'symbol': symbol, 'interval': interval,
                  'limit': 1000, 'startTime': current_start}
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            klines.extend(data)
            current_start = data[-1][0] + 1
            if len(data) < 1000:
                break
        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
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


# ── ICT Feature Calculation ─────────────────────────────────────────────────
def calculate_ict_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Tính toàn bộ ICT features + benchmark indicators trên DataFrame có OHLCV.
    """
    # ICT core features
    df = compute_all_ict_features(df)

    # Benchmark indicators (không dùng trong model, chỉ cho visualization)
    periods = _ema_periods(tf)
    df['rsi']      = ta.momentum.rsi(df['close'], window=14)
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=periods['fast'])
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=periods['slow'])

    return df


# ── ICT Scoring ─────────────────────────────────────────────────────────────
def _ict_long_score(f: dict) -> tuple:
    score, max_pts, reasons = 0.0, 0.80, []
    if f.get('htf_trend', 0) == 1:
        score += 0.20; reasons.append("HTF bullish")
    bisi = f.get('inside_bisi_mid_zone', 0)
    if bisi >= 0.7:
        score += 0.20; reasons.append("BISI midpoint reaction")
    elif bisi >= 0.4:
        score += 0.10
    mss = f.get('mss_strength', 0)
    if mss >= 0.5:
        score += 0.15; reasons.append("Bullish MSS")
    elif mss >= 0.2:
        score += 0.07
    if f.get('liquidity_sweep_detected', 0) == 1:
        score += 0.15; reasons.append("Liq sweep low ✓")
    if f.get('cisd_state', 0) == 1:
        score += 0.10; reasons.append("Bullish CISD")
    base = score / max_pts
    macro, session = f.get('macro_window', 0), f.get('session_score', 0.3)
    if macro == 2:
        mult = 1.0 + session * 0.30
        if session >= 0.9: reasons.append("NY Macro ⚡")
    elif macro == 1:
        mult = 1.0 + session * 0.15; reasons.append("London Macro")
    else:
        mult = 0.70 + session * 0.10
    return min(base * mult, 1.0), reasons


def _ict_short_score(f: dict) -> tuple:
    score, max_pts, reasons = 0.0, 0.80, []
    if f.get('htf_trend', 0) == -1:
        score += 0.20; reasons.append("HTF bearish")
    sibi = f.get('inside_sibi_mid_zone', 0)
    if sibi >= 0.7:
        score += 0.20; reasons.append("SIBI midpoint reaction")
    elif sibi >= 0.4:
        score += 0.10
    mss = f.get('mss_strength', 0)
    if mss <= -0.5:
        score += 0.15; reasons.append("Bearish MSS")
    elif mss <= -0.2:
        score += 0.07
    if f.get('liquidity_sweep_detected', 0) == 2:
        score += 0.15; reasons.append("Liq sweep high ✓")
    if f.get('cisd_state', 0) == -1:
        score += 0.10; reasons.append("Bearish CISD")
    base = score / max_pts
    macro, session = f.get('macro_window', 0), f.get('session_score', 0.3)
    if macro == 2:
        mult = 1.0 + session * 0.30
        if session >= 0.9: reasons.append("NY Macro ⚡")
    elif macro == 1:
        mult = 1.0 + session * 0.15
    else:
        mult = 0.70 + session * 0.10
    return min(base * mult, 1.0), reasons


def get_htf_alignment_trend(tf: str) -> int:
    htf_map = {
        '1min': '15min',
        '5min': '1H',
        '15min': '4H',
        '1H': '1D',
        '4H': '1W',
        '1D': '1M'
    }
    htf = htf_map.get(tf)
    if not htf: return 0
    
    # 1. Fast path for existing features
    if htf in TIMEFRAMES:
        local_file = f'data/btc_features_{htf}.parquet'
        if os.path.exists(local_file):
            try:
                df = pd.read_parquet(local_file)
                if not df.empty and 'htf_trend' in df.columns:
                    return int(df['htf_trend'].iloc[-1])
            except:
                pass
        return 0
    
    # 2. On the fly calculation for 1W, 1M from 1D data
    local_file = 'data/btc_features_1D.parquet'
    if os.path.exists(local_file):
        try:
            df = pd.read_parquet(local_file)
            if not df.empty:
                df_res = df.set_index('timestamp').resample(htf).agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                }).dropna().reset_index()
                if len(df_res) > 2:
                    from ict_features import compute_htf_trend
                    # Use smaller lookback for 1W/1M
                    lookback = min(20, max(2, len(df_res) // 2))
                    h = df_res['high']
                    l = df_res['low']
                    prev_high_max = h.shift(1).rolling(lookback, min_periods=max(1, lookback//2)).max()
                    prev_low_min  = l.shift(1).rolling(lookback, min_periods=max(1, lookback//2)).min()
                    prev_low_max  = l.shift(1).rolling(lookback, min_periods=max(1, lookback//2)).max()
                    
                    hh = h > prev_high_max
                    hl = l > (prev_low_min + (prev_low_max - prev_low_min) * 0.3)
                    lh = h < prev_high_max
                    ll = l < prev_low_min
                    
                    trend = pd.Series(0, index=df_res.index, dtype=int)
                    trend[hh & hl] = 1
                    trend[lh & ll] = -1
                    return int(trend.iloc[-1])
        except:
            pass
    return 0

def run_ict_predict(df: pd.DataFrame, tf: str, htf_trend_val: int = 0) -> pd.DataFrame:
    """Apply ICT scoring + XGBoost to each row of df."""
    thresh    = THRESHOLDS.get(tf, {"buy": 0.65, "sell": 0.35})
    buy_thr   = thresh.get("buy",  0.65)
    sell_thr  = thresh.get("sell", 0.35)

    # XGBoost bulk predict
    xgb_probs = np.full(len(df), 0.5)
    if tf in MODELS:
        feat_cols = [c for c in ICT_FEATURES if c in df.columns]
        try:
            xgb_probs = MODELS[tf].predict_proba(df[feat_cols].fillna(0))[:, 1]
        except Exception as e:
            logger.warning(f"XGBoost batch predict error: {e}")

    biases, probs_up, probs_down, confs, contexts = [], [], [], [], []

    for i, row in enumerate(df.to_dict('records')):
        long_s, long_r  = _ict_long_score(row)
        short_s, short_r = _ict_short_score(row)

        pu = float(np.clip(0.60 * xgb_probs[i] + 0.40 * long_s,  0, 1))
        pd_ = float(np.clip(0.60 * (1 - xgb_probs[i]) + 0.40 * short_s, 0, 1))
        tot = pu + pd_
        if tot > 0:
            pu /= tot; pd_ /= tot

        conf = float(abs(pu - 0.5) * 2)

        if pu >= buy_thr:
            if htf_trend_val == 1:
                bias, reasons = "STRONG_LONG", long_r + ["Aligned with HTF Bullish"]
            else:
                bias, reasons = "LONG",  long_r
        elif pd_ >= (1 - sell_thr):
            if htf_trend_val == -1:
                bias, reasons = "STRONG_SHORT", short_r + ["Aligned with HTF Bearish"]
            else:
                bias, reasons = "SHORT", short_r
        else:
            bias, reasons = "NEUTRAL", ["Insufficient confluence"]

        macro_lbl = MACRO_LABEL.get(int(row.get('macro_window', 0)), "")
        display_bias = bias.replace('_', ' ')
        ctx = f"{macro_lbl + ' | ' if macro_lbl else ''}{display_bias}: {', '.join(reasons)}"

        biases.append(bias); probs_up.append(pu); probs_down.append(pd_)
        confs.append(conf);  contexts.append(ctx)

    df['bias']             = biases
    df['probability_up']   = probs_up
    df['probability_down'] = probs_down
    df['confidence_score'] = confs
    df['ict_context']      = contexts
    df['signal'] = df['bias']   # LONG / SHORT / NEUTRAL
    df['prob']   = df['probability_up']
    return df


# ── HDFS read ───────────────────────────────────────────────────────────────
def read_features_from_hdfs(timeframe: str) -> pd.DataFrame:
    try:
        hdfs_path = f'/user/hdoop/bigdata/features/btc_features_{timeframe}.parquet'
        tmp_file  = f'/tmp/btc_features_{timeframe}_hdfs.parquet'
        result = subprocess.run(
            ['hdfs', 'dfs', '-get', '-f', hdfs_path, tmp_file],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            df = pd.read_parquet(tmp_file)
            logger.info(f"[{timeframe}] {len(df)} rows from HDFS")
            return df
    except Exception as e:
        logger.warning(f"[{timeframe}] HDFS fallback failed: {e}")
    raise FileNotFoundError(f"Data for {timeframe} not found locally or on HDFS")


import subprocess


def extract_overlays(df: pd.DataFrame, n: int = 200) -> dict:
    """
    Extract ICT overlay data for chart visualization.
    Uses the last `n` rows to find active FVG zones, MSS, CISD, Liquidity events.
    """
    from ict_features import _detect_fvg_vectors

    sub = df.tail(n).reset_index(drop=True)

    # ── FVG Zones ──────────────────────────────────────────────────
    bfl, bfh, berfl, berfh = _detect_fvg_vectors(sub)
    fvg_zones = []

    # Bullish FVGs (last 8 unfilled preferred)
    bull_idx = sub.index[bfl.notna()].tolist()
    for idx in bull_idx[-8:]:
        fl, fh = float(bfl.iloc[idx]), float(bfh.iloc[idx])
        fill  = float(sub['fvg_fill_ratio'].iloc[idx]) if 'fvg_fill_ratio' in sub.columns else 0.0
        fvg_zones.append({
            'type': 'bull', 'filled': fill >= 0.95,
            'time': int(sub['timestamp'].iloc[idx].timestamp()),
            'fvg_high': fh, 'fvg_low': fl,
            'fvg_mid': round((fh + fl) / 2, 2),
            'fill_ratio': round(fill, 3),
        })

    # Bearish FVGs
    bear_idx = sub.index[berfl.notna()].tolist()
    for idx in bear_idx[-8:]:
        fl, fh = float(berfl.iloc[idx]), float(berfh.iloc[idx])
        fvg_zones.append({
            'type': 'bear', 'filled': False,
            'time': int(sub['timestamp'].iloc[idx].timestamp()),
            'fvg_high': fh, 'fvg_low': fl,
            'fvg_mid': round((fh + fl) / 2, 2),
            'fill_ratio': 0.0,
        })

    # ── MSS Events ─────────────────────────────────────────────────
    mss_events = []
    if 'mss_strength' in sub.columns:
        for _, row in sub[sub['mss_strength'].abs() > 0.3].tail(30).iterrows():
            mss_events.append({
                'time':      int(row['timestamp'].timestamp()),
                'direction': 'bull' if row['mss_strength'] > 0 else 'bear',
                'strength':  round(float(abs(row['mss_strength'])), 3),
            })

    # ── CISD Events ────────────────────────────────────────────────
    cisd_events = []
    if 'cisd_state' in sub.columns:
        for _, row in sub[sub['cisd_state'] != 0].tail(20).iterrows():
            cisd_events.append({
                'time':  int(row['timestamp'].timestamp()),
                'state': int(row['cisd_state']),
            })

    # ── Liquidity Sweeps ───────────────────────────────────────────
    liq_events = []
    if 'liquidity_sweep_detected' in sub.columns:
        for _, row in sub[sub['liquidity_sweep_detected'] != 0].tail(20).iterrows():
            liq_events.append({
                'time': int(row['timestamp'].timestamp()),
                'type': int(row['liquidity_sweep_detected']),  # 1=sweep_low, 2=sweep_high
            })

    return {
        'fvg_zones':   fvg_zones,
        'mss_events':  mss_events,
        'cisd_events': cisd_events,
        'liq_events':  liq_events,
    }


# ── API Routes ───────────────────────────────────────────────────────────────
@app.route('/api/hdfs-status', methods=['GET'])
def hdfs_status():
    try:
        result = subprocess.run(
            ['hdfs', 'dfs', '-ls', '-h', '/user/hdoop/bigdata/'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return jsonify({"status": "connected", "content": result.stdout})
        return jsonify({"status": "error", "message": result.stderr})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/data', methods=['GET'])
def get_data():
    timeframe = request.args.get('timeframe', '1min')
    if timeframe not in TIMEFRAMES:
        timeframe = '1min'

    try:
        df = read_features_from_hdfs(timeframe)
        df = df.tail(1000).copy()

        last_ts  = int(df['timestamp'].max().timestamp() * 1000)
        interval = BINANCE_INTERVALS[timeframe]
        new_candles = fetch_missing_candles('BTCUSDT', interval, last_ts + 1)

        if not new_candles.empty:
            logger.info(f"Patched {len(new_candles)} candles for {timeframe}")
            
            hist_raw = pd.DataFrame({
                'timestamp': df['timestamp'].values.astype('datetime64[us]'),
                'open':  df['open'].values.astype(float),
                'high':  df['high'].values.astype(float),
                'low':   df['low'].values.astype(float),
                'close': df['close'].values.astype(float),
                'volume':df['volume'].values.astype(float),
            })
            
            new_clean = pd.DataFrame({
                'timestamp': new_candles['timestamp'].values.astype('datetime64[us]'),
                'open':  new_candles['open'].values.astype(float),
                'high':  new_candles['high'].values.astype(float),
                'low':   new_candles['low'].values.astype(float),
                'close': new_candles['close'].values.astype(float),
                'volume':new_candles['volume'].values.astype(float),
            })
            merged = pd.concat([hist_raw, new_clean], ignore_index=True)
            merged.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            merged = merged.sort_values('timestamp').reset_index(drop=True)
            df = calculate_ict_features(merged, timeframe)

        # Apply ICT predict
        htf_trend = get_htf_alignment_trend(timeframe)
        df = run_ict_predict(df, timeframe, htf_trend)

        df['time'] = df['timestamp'].apply(lambda x: int(x.timestamp()))
        df = df.replace({np.nan: None})

        # Build response columns
        base_cols = ['time', 'open', 'high', 'low', 'close', 'volume',
                     'bias', 'probability_up', 'probability_down',
                     'ict_context',
                     'signal', 'prob',  # legacy compat
                     'htf_trend', 'mss_strength',
                     'inside_bisi_mid_zone', 'inside_sibi_mid_zone',
                     'liquidity_sweep_detected', 'macro_window']
        bench_cols = [c for c in ['ema_fast', 'ema_slow', 'rsi'] if c in df.columns]
        out_cols   = [c for c in base_cols + bench_cols if c in df.columns]

        result = df[out_cols].to_dict(orient='records')

        # Market context (latest row)
        latest = df.iloc[-1]
        macro_val = int(latest.get('macro_window', 0)) if pd.notna(latest.get('macro_window')) else 0
        htf_val   = int(latest.get('htf_trend', 0))   if pd.notna(latest.get('htf_trend'))   else 0

        market_context = {
            "htf_bias":    HTF_LABEL.get(htf_val, "RANGING"),
            "killzone":    MACRO_LABEL.get(macro_val, "Outside Macro"),
            "macro_active": macro_val > 0,
        }

        # Extract ICT overlays for chart visualization
        overlays = extract_overlays(df, n=200)

        return jsonify({
            "data":           result,
            "market_context": market_context,
            "thresholds":     THRESHOLDS.get(timeframe, {"buy": 0.65, "sell": 0.35}),
            "ema_periods":    _ema_periods(timeframe),
            "overlays":       overlays,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
