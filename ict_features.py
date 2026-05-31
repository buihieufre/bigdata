"""
ICT / Smart Money Concept Feature Engine — VECTORIZED
======================================================
Fully vectorized with NumPy/Pandas rolling — O(n) not O(n²).
Handles 1.5M+ rows in seconds.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. HTF TREND
# ─────────────────────────────────────────────
def compute_htf_trend(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    h = df['high']
    l = df['low']
    prev_high_max = h.shift(1).rolling(lookback, min_periods=lookback).max()
    prev_low_min  = l.shift(1).rolling(lookback, min_periods=lookback).min()
    prev_low_max  = l.shift(1).rolling(lookback, min_periods=lookback).max()

    hh = h > prev_high_max
    hl = l > (prev_low_min + (prev_low_max - prev_low_min) * 0.3)
    lh = h < prev_high_max
    ll = l < prev_low_min

    trend = pd.Series(0, index=df.index, dtype=int)
    trend[hh & hl] = 1
    trend[lh & ll] = -1
    return trend


def compute_structure_strength(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    h = df['high']
    l = df['low']
    prev_high_max = h.shift(1).rolling(lookback, min_periods=lookback).max()
    prev_low_min  = l.shift(1).rolling(lookback, min_periods=lookback).min()
    prev_low_med  = l.shift(1).rolling(lookback, min_periods=lookback).median()
    prev_high_med = h.shift(1).rolling(lookback, min_periods=lookback).median()

    hh = (h > prev_high_max).astype(float)
    hl = (l > prev_low_med).astype(float)
    lh = (h < prev_high_med).astype(float)
    ll = (l < prev_low_min).astype(float)

    bull = (hh + hl) / 2
    bear = (lh + ll) / 2
    return bull.combine(bear, max).fillna(0)


# ─────────────────────────────────────────────
# 2. DISPLACEMENT
# ─────────────────────────────────────────────
def compute_displacement_strength(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
    body = (df['close'] - df['open']).abs()
    rng  = (df['high'] - df['low']).replace(0, np.nan)
    body_ratio   = (body / rng).clip(0, 1)
    atr          = rng.rolling(atr_window, min_periods=1).mean()
    atr_expansion = (rng / atr.replace(0, np.nan)).clip(0, 3) / 3
    return (body_ratio * atr_expansion).clip(0, 1).fillna(0)


# ─────────────────────────────────────────────
# 3. MSS
# ─────────────────────────────────────────────
def compute_mss(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    h = df['high']
    l = df['low']
    c = df['close']
    rng  = (h - l).replace(0, np.nan)
    body = (c - df['open']).abs()
    body_ratio = (body / rng).clip(0, 1).fillna(0)

    atr = rng.rolling(14, min_periods=1).mean()
    expansion = (rng / atr.replace(0, np.nan)).clip(0, 2) / 2

    swing_high = h.shift(1).rolling(lookback, min_periods=lookback).max()
    swing_low  = l.shift(1).rolling(lookback, min_periods=lookback).min()

    bull_break = (c > swing_high) & (body_ratio > 0.55)
    bear_break = (c < swing_low)  & (body_ratio > 0.55)

    strength = (body_ratio * expansion.fillna(0)).clip(0, 1)
    mss = pd.Series(0.0, index=df.index)
    mss[bull_break] =  strength[bull_break]
    mss[bear_break] = -strength[bear_break]
    return mss.clip(-1, 1)


# ─────────────────────────────────────────────
# 4. CISD
# ─────────────────────────────────────────────
def compute_cisd(df: pd.DataFrame, window: int = 5) -> pd.Series:
    rng  = (df['high'] - df['low']).replace(0, np.nan)
    body = df['close'] - df['open']
    body_ratio = (body / rng).fillna(0)
    atr = rng.rolling(14, min_periods=1).mean()
    expansion = rng > (atr * 1.2)

    bull_run = (body_ratio > 0.5).rolling(window, min_periods=window).sum() >= (window - 1)
    bear_run = (body_ratio < -0.5).rolling(window, min_periods=window).sum() >= (window - 1)

    cisd = pd.Series(0, index=df.index, dtype=int)
    cisd[bull_run & expansion] =  1
    cisd[bear_run & expansion] = -1
    return cisd


# ─────────────────────────────────────────────
# 5. FVG — fully vectorized
# ─────────────────────────────────────────────
def _detect_fvg_vectors(df: pd.DataFrame):
    """Returns bull/bear FVG high/low series using shift-based vectorization."""
    h2 = df['high'].shift(2)
    l2 = df['low'].shift(2)
    l0 = df['low']
    h0 = df['high']

    bull_mask = l0 > h2
    bear_mask = h0 < l2

    bull_fvg_low  = h2.where(bull_mask)   # gap from h[i-2] to l[i]
    bull_fvg_high = l0.where(bull_mask)
    bear_fvg_low  = h0.where(bear_mask)   # gap from h[i] to l[i-2]
    bear_fvg_high = l2.where(bear_mask)

    return bull_fvg_low, bull_fvg_high, bear_fvg_low, bear_fvg_high


# ─────────────────────────────────────────────
# 6. BISI / SIBI — vectorized
# ─────────────────────────────────────────────
def compute_bisi_sibi(df: pd.DataFrame):
    bfl, bfh, berfl, berfh = _detect_fvg_vectors(df)
    c = df['close']

    # Forward-fill
    bfl_ff   = bfl.ffill()
    bfh_ff   = bfh.ffill()
    berfl_ff = berfl.ffill()
    berfh_ff = berfh.ffill()

    # ── BISI (bullish FVG) ──────────────────────────────────────
    bull_mid    = (bfl_ff + bfh_ff) / 2
    bull_width  = (bfh_ff - bfl_ff).clip(lower=1e-10)
    upper_25    = bull_mid + bull_width * 0.25

    in_fvg_bull   = (c >= bfl_ff) & (c <= bfh_ff) & bfl_ff.notna()
    in_upper_bull = in_fvg_bull & (c >= bull_mid) & (c <= upper_25)
    in_lower_bull = in_fvg_bull & (c >= bfl_ff)   & (c < bull_mid)

    bisi = pd.Series(0.0, index=df.index)
    bisi[in_upper_bull] = 1.0
    lower_score_bull = ((c - bfl_ff) / (bull_mid - bfl_ff).clip(lower=1e-10) * 0.5).clip(0, 0.5)
    bisi[in_lower_bull] = lower_score_bull[in_lower_bull]

    # ── SIBI (bearish FVG) ──────────────────────────────────────
    bear_mid    = (berfl_ff + berfh_ff) / 2
    bear_width  = (berfh_ff - berfl_ff).clip(lower=1e-10)
    lower_25    = bear_mid - bear_width * 0.25

    in_fvg_bear   = (c >= berfl_ff) & (c <= berfh_ff) & berfl_ff.notna()
    in_lower_bear = in_fvg_bear & (c >= lower_25) & (c <= bear_mid)
    in_upper_bear = in_fvg_bear & (c > bear_mid)  & (c <= berfh_ff)

    sibi = pd.Series(0.0, index=df.index)
    sibi[in_lower_bear] = 1.0
    upper_score_bear = ((berfh_ff - c) / (berfh_ff - bear_mid).clip(lower=1e-10) * 0.5).clip(0, 0.5)
    sibi[in_upper_bear] = upper_score_bear[in_upper_bear]

    return bisi.fillna(0), sibi.fillna(0)


def compute_imbalance_reaction(bisi: pd.Series, sibi: pd.Series) -> pd.Series:
    return (bisi - sibi).clip(-1, 1)


# ─────────────────────────────────────────────
# 7. LIQUIDITY — vectorized
# ─────────────────────────────────────────────
def detect_liquidity(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    h = df['high']
    l = df['low']
    c = df['close']

    prev_high = h.shift(1).rolling(lookback, min_periods=lookback).max()
    prev_low  = l.shift(1).rolling(lookback, min_periods=lookback).min()

    sweep_low  = (l < prev_low)  & (c > prev_low)
    sweep_high = (h > prev_high) & (c < prev_high)

    liq = pd.Series(0, index=df.index, dtype=int)
    liq[sweep_low]  = 1
    liq[sweep_high] = 2
    return liq



# ─────────────────────────────────────────────
# 9. SESSION / MACRO SCORE — vectorized
# ─────────────────────────────────────────────
def compute_session_score(timestamps: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamps)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    h = ts.dt.hour
    m = ts.dt.minute

    score = pd.Series(0.10, index=timestamps.index)

    # Asian session
    score[(h >= 0) & (h < 7)] = 0.20
    # London (no macro)
    score[(h >= 7) & (h < 12)] = 0.50
    # NY active (no macro)
    score[(h >= 13) & (h < 20)] = 0.65

    # ── London Macros ───────────────────────────────────────────
    score[((h == 11) & (m >= 50)) | ((h == 12) & (m <= 10))] = 0.75  # London Close
    score[((h == 9)  & (m >= 50)) | ((h == 10) & (m <= 10))] = 0.80  # London Mid
    score[((h == 6)  & (m >= 50)) | ((h == 7)  & (m <= 10))] = 0.90  # London Open

    # ── NY Macros (applied last → highest priority) ─────────────
    score[((h == 15) & (m >= 50)) | ((h == 16) & (m <= 10))] = 0.80  # NY Lunch
    score[((h == 19) & (m >= 50)) | ((h == 20) & (m <= 10))] = 0.85  # NY Close
    score[((h == 16) & (m >= 50)) | ((h == 17) & (m <= 10))] = 0.85  # NY Afternoon
    score[((h == 14) & (m >= 50)) | ((h == 15) & (m <= 10))] = 0.85  # NY Mid-Morning
    score[((h == 18) & (m >= 50)) | ((h == 19) & (m <= 10))] = 0.90  # NY Power Hour
    score[((h == 12) & (m >= 50)) | ((h == 13) & (m <= 10))] = 0.90  # NY Pre-Market
    score[((h == 13) & (m >= 50)) | ((h == 14) & (m <= 10))] = 1.00  # NY Open ⭐
    score[((h == 17) & (m >= 50)) | ((h == 18) & (m <= 10))] = 1.00  # NY PM Open ⭐

    return score


def compute_macro_window(timestamps: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamps)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    h = ts.dt.hour
    m = ts.dt.minute

    in_macro = (m >= 50) | (m <= 10)

    NY_MACRO_H  = (h >= 12) & (h <= 20)
    LON_MACRO_H = ((h >= 6) & (h <= 7)) | ((h >= 9) & (h <= 12))

    macro = pd.Series(0, index=timestamps.index, dtype=int)
    macro[in_macro & LON_MACRO_H] = 1
    macro[in_macro & NY_MACRO_H]  = 2   # NY overwrites London overlap
    return macro



# ─────────────────────────────────────────────
# 8. MASTER FUNCTION
# ─────────────────────────────────────────────
def compute_all_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 11 ICT features on DataFrame with columns:
      open, high, low, close, volume, timestamp

    Fully vectorized — suitable for 1M+ rows.
    """
    df = df.copy()

    df['htf_trend']             = compute_htf_trend(df)
    df['structure_strength']    = compute_structure_strength(df)
    df['displacement_strength'] = compute_displacement_strength(df)
    df['mss_strength']          = compute_mss(df)
    df['cisd_state']            = compute_cisd(df)

    bisi, sibi = compute_bisi_sibi(df)
    df['inside_bisi_mid_zone']     = bisi
    df['inside_sibi_mid_zone']     = sibi
    df['imbalance_reaction_score'] = compute_imbalance_reaction(bisi, sibi)

    df['liquidity_sweep_detected'] = detect_liquidity(df)
    df['session_score']            = compute_session_score(df['timestamp'])
    df['macro_window']             = compute_macro_window(df['timestamp'])

    return df


# ─────────────────────────────────────────────
# ICT FEATURES LIST (11 features)
# ─────────────────────────────────────────────
ICT_FEATURES = [
    'htf_trend', 'structure_strength', 'displacement_strength',
    'mss_strength', 'cisd_state',
    'inside_bisi_mid_zone', 'inside_sibi_mid_zone', 'imbalance_reaction_score',
    'liquidity_sweep_detected', 'session_score', 'macro_window',
]
