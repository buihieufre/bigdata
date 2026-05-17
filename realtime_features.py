import pandas as pd
import numpy as np
import ta

class RealtimeFeatureGenerator:
    """
    Stateful feature generator for streaming data.
    Maintains a rolling window of recent prices.
    """
    WINDOW_SIZE = 250
    history = []
    
    @classmethod
    def update(cls, close: float, volume: float) -> pd.DataFrame:
        """
        Add new price point and calculate features if window is full
        """
        cls.history.append({'close': close, 'volume': volume})
        
        # Maintain window size
        if len(cls.history) > cls.WINDOW_SIZE:
            cls.history.pop(0)
            
        # Only generate features if we have enough data
        if len(cls.history) == cls.WINDOW_SIZE:
            df = pd.DataFrame(cls.history)
            
            # Calculate features (same as training)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=21)
            
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            
            df['return_1'] = df['close'].pct_change(1)
            df['return_5'] = df['close'].pct_change(5)
            df['volatility'] = df['return_1'].rolling(window=20).std()
            
            # Return only the latest row's features
            features = [
                'rsi', 'ema_fast', 'ema_slow', 'macd', 'macd_signal', 
                'bb_high', 'bb_low', 'return_1', 'return_5', 'volatility', 'volume'
            ]
            
            latest_features = df[features].iloc[-1:]
            
            # Check for NaNs
            if latest_features.isnull().values.any():
                return None
                
            return latest_features
            
        return None
