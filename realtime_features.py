"""
Realtime ICT Feature Generator
================================
Stateful generator dùng cho Spark Structured Streaming.
Maintains a rolling window of 1000 1-min candles (~16h40m)
để đảm bảo đủ context xem cấu trúc Macro liên phiên.
"""

import pandas as pd
import numpy as np
from ict_features import compute_all_ict_features, ICT_FEATURES


class RealtimeICTFeatureGenerator:
    """
    Stateful ICT feature generator cho streaming data.
    Cần full OHLCV + timestamp (khác với version cũ chỉ dùng close/volume).

    Window 1000 candles (1-min) = ~16h40m:
      - Đủ thấy Asian → London → NY session structure
      - Đủ detect BISI/SIBI từ session trước
      - Đủ detect Macro Windows xx:50→xx:10
    """

    WINDOW_SIZE = 1000   # 1000 × 1-min = 16h40m
    MIN_WINDOW  = 50     # Minimum để tính ICT (swing detection cần ít nhất 50)

    history: list = []

    @classmethod
    def reset(cls):
        cls.history = []

    @classmethod
    def update(cls,
               open_: float,
               high: float,
               low: float,
               close: float,
               volume: float,
               timestamp_ms: int) -> pd.DataFrame | None:
        """
        Thêm candle mới vào rolling window và tính ICT features.

        Args:
            open_, high, low, close, volume: OHLCV values
            timestamp_ms: kline open time từ Binance (milliseconds UTC)

        Returns:
            pd.DataFrame với 1 row = ICT features của candle hiện tại,
            hoặc None nếu chưa đủ MIN_WINDOW candles.
        """
        ts_utc = pd.Timestamp(timestamp_ms, unit='ms', tz='UTC').tz_localize(None)

        cls.history.append({
            'open':      float(open_),
            'high':      float(high),
            'low':       float(low),
            'close':     float(close),
            'volume':    float(volume),
            'timestamp': ts_utc,
        })

        # Maintain rolling window
        if len(cls.history) > cls.WINDOW_SIZE:
            cls.history.pop(0)

        # Need minimum window to compute meaningful ICT features
        if len(cls.history) < cls.MIN_WINDOW:
            return None

        df = pd.DataFrame(cls.history)

        try:
            df_ict = compute_all_ict_features(df)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"ICT compute error: {e}")
            return None

        # Return only the latest row's features
        latest = df_ict[ICT_FEATURES].iloc[-1:]

        # Safety: drop if any core feature is NaN
        if latest.isnull().values.any():
            return None

        return latest
