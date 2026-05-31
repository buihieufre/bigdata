"""
Realtime ICT Predictor
=======================
ICT/SMC scoring engine kết hợp XGBoost + rule-based ICT logic.

Output:
  bias            : LONG / SHORT / NEUTRAL
  probability_up  : float [0,1]
  probability_down: float [0,1]
  ict_context     : human-readable string mô tả lý do

NY Session Macro Windows (xx:50→xx:10) nhận hệ số nhân cao nhất.
"""

import joblib
import pandas as pd
import numpy as np
import os
import logging

from ict_features import ICT_FEATURES

logger = logging.getLogger(__name__)


class RealtimeICTPredictor:
    MODELS     : dict = {}
    THRESHOLDS : dict = {}
    TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']

    # ── Model loading ─────────────────────────────────────────────
    @classmethod
    def load_models(cls):
        if cls.MODELS:
            return
        import json
        thresh_file = 'models/thresholds.json'
        if os.path.exists(thresh_file):
            with open(thresh_file) as f:
                cls.THRESHOLDS = json.load(f)

        for tf in cls.TIMEFRAMES:
            # Try new ICT model first, fall back to old model name
            for name in [f'models/xgb_ict_{tf}.pkl', f'models/xgb_model_{tf}.pkl']:
                if os.path.exists(name):
                    cls.MODELS[tf] = joblib.load(name)
                    logger.info(f"Loaded model: {name}")
                    break

    # ── ICT Rule-Based Score ──────────────────────────────────────
    @classmethod
    def _ict_long_score(cls, f: dict) -> tuple[float, list]:
        """
        Returns (long_score [0,1], reasons list).
        Scoring tiers theo ICT confluence:
          Tier 1 – Structure (20%)
          Tier 2 – Zone reaction BISI (20%)
          Tier 3 – MSS (15%)
          Tier 4 – Liquidity sweep low (15%)
          Tier 5 – CISD bullish (10%)
          Tier 6 – Momentum (10%)
          Tier 7 – Session/Macro multiplier (up to ×1.30 for NY macro)
        """
        score   = 0.0
        max_pts = 0.90
        reasons = []

        # T1 – HTF trend
        if f.get('htf_trend', 0) == 1:
            score += 0.20
            reasons.append("HTF bullish")

        # T2 – BISI zone
        bisi = f.get('inside_bisi_mid_zone', 0)
        if bisi >= 0.7:
            score += 0.20
            reasons.append("Inside BISI midpoint (strong)")
        elif bisi >= 0.4:
            score += 0.10
            reasons.append("Inside BISI zone")

        # T3 – MSS strength
        mss = f.get('mss_strength', 0)
        if mss >= 0.5:
            score += 0.15
            reasons.append(f"Bullish MSS ({mss:.2f})")
        elif mss >= 0.2:
            score += 0.07

        # T4 – Liquidity sweep low
        if f.get('liquidity_sweep_detected', 0) == 1:
            score += 0.15
            reasons.append("Liquidity sweep low (bullish)")

        # T5 – CISD
        if f.get('cisd_state', 0) == 1:
            score += 0.10
            reasons.append("Bullish CISD expansion")

        # T6 – Momentum
        mom = f.get('momentum_score', 0)
        if mom >= 0.5:
            score += 0.10
            reasons.append(f"Momentum expansion ({mom:.2f})")
        elif mom >= 0.2:
            score += 0.05

        base_ratio = score / max_pts

        # T7 – Macro multiplier (Session score removed)
        macro   = f.get('macro_window', 0)
        if macro == 2:           # NY macro → highest boost
            multiplier = 1.30
            reasons.append("NY Macro Window ⚡")
        elif macro == 1:         # London macro
            multiplier = 1.15
            reasons.append("London Macro Window")
        else:                    # No macro
            multiplier = 1.0

        final = min(base_ratio * multiplier, 1.0)
        return final, reasons

    @classmethod
    def _ict_short_score(cls, f: dict) -> tuple[float, list]:
        """Mirror of long score for bearish bias."""
        score   = 0.0
        max_pts = 0.90
        reasons = []

        if f.get('htf_trend', 0) == -1:
            score += 0.20
            reasons.append("HTF bearish")

        sibi = f.get('inside_sibi_mid_zone', 0)
        if sibi >= 0.7:
            score += 0.20
            reasons.append("Inside SIBI midpoint (strong)")
        elif sibi >= 0.4:
            score += 0.10

        mss = f.get('mss_strength', 0)
        if mss <= -0.5:
            score += 0.15
            reasons.append(f"Bearish MSS ({mss:.2f})")
        elif mss <= -0.2:
            score += 0.07

        if f.get('liquidity_sweep_detected', 0) == 2:
            score += 0.15
            reasons.append("Liquidity sweep high (bearish)")

        if f.get('cisd_state', 0) == -1:
            score += 0.10
            reasons.append("Bearish CISD expansion")

        mom = f.get('momentum_score', 0)
        if mom <= -0.5:
            score += 0.10
            reasons.append(f"Bearish momentum ({mom:.2f})")
        elif mom <= -0.2:
            score += 0.05

        base_ratio = score / max_pts

        macro   = f.get('macro_window', 0)
        if macro == 2:
            multiplier = 1.30
            reasons.append("NY Macro Window ⚡")
        elif macro == 1:
            multiplier = 1.15
        else:
            multiplier = 1.0

        final = min(base_ratio * multiplier, 1.0)
        return final, reasons

    # ── Main predict ──────────────────────────────────────────────
    @classmethod
    def predict(cls, features: pd.DataFrame, tf: str = '1min') -> dict:
        cls.load_models()

        f = features.iloc[0].to_dict()

        # ── XGBoost probability ──────────────────────────────────
        xgb_prob = 0.5
        if tf in cls.MODELS:
            try:
                feat_cols = [c for c in ICT_FEATURES if c in features.columns]
                xgb_prob  = cls.MODELS[tf].predict_proba(
                    features[feat_cols].fillna(0)
                )[0][1]
            except Exception as e:
                logger.warning(f"XGBoost predict failed: {e}")

        # ── Rule-based ICT scores ────────────────────────────────
        long_score,  long_reasons  = cls._ict_long_score(f)
        short_score, short_reasons = cls._ict_short_score(f)

        # ── Combine: 60% XGB + 40% ICT rules ────────────────────
        prob_up   = 0.60 * xgb_prob + 0.40 * long_score
        prob_down = 0.60 * (1 - xgb_prob) + 0.40 * short_score

        # Normalise so they sum ≈ 1
        total = prob_up + prob_down
        if total > 0:
            prob_up   /= total
            prob_down /= total

        prob_up   = float(np.clip(prob_up,   0, 1))
        prob_down = float(np.clip(prob_down, 0, 1))

        # ── Thresholds ────────────────────────────────────────────
        thresh     = cls.THRESHOLDS.get(tf, {"buy": 0.65, "sell": 0.35})
        buy_thr    = thresh.get("buy",  0.65)
        sell_thr   = thresh.get("sell", 0.35)

        if prob_up >= buy_thr:
            bias    = "LONG"
            reasons = long_reasons
        elif prob_down >= (1 - sell_thr):
            bias    = "SHORT"
            reasons = short_reasons
        else:
            bias    = "NEUTRAL"
            reasons = ["Insufficient confluence"]

        # ── Human-readable context ────────────────────────────────
        macro_label = {0: "", 1: "London Macro", 2: "NY Macro ⚡"}.get(
            int(f.get('macro_window', 0)), "")
        context = f"{bias} | {', '.join(reasons)}"
        if macro_label:
            context = f"{macro_label} | {context}"

        return {
            "bias":             bias,
            "probability_up":   prob_up,
            "probability_down": prob_down,
            "ict_context":      context,
            # legacy aliases for backward compat
            "signal":           bias,
            "prob":             prob_up,
        }
