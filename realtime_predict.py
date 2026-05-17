import joblib
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class RealtimePredictor:
    """
    Loads the trained models and performs inference on streaming features based on timeframe.
    """
    MODELS = {}
    TIMEFRAMES = ['1min', '5min', '15min', '1H', '4H', '1D']
    
    @classmethod
    def load_models(cls):
        if not cls.MODELS:
            for tf in cls.TIMEFRAMES:
                model_file = f'models/xgb_model_{tf}.pkl'
                if os.path.exists(model_file):
                    cls.MODELS[tf] = joblib.load(model_file)
                    logger.info(f"Model for {tf} loaded successfully.")
                else:
                    logger.warning(f"Model for {tf} not found at {model_file}.")
            
    @classmethod
    def predict(cls, features: pd.DataFrame, tf: str = '1min') -> dict:
        """
        Predict based on features and timeframe
        """
        cls.load_models()
        
        if tf not in cls.MODELS:
            logger.error(f"No loaded model for timeframe: {tf}")
            return {"signal": "HOLD", "prob": 0.5}
            
        model = cls.MODELS[tf]
        
        # Predict probability
        prob = model.predict_proba(features)[0][1]
        
        signal = 'HOLD'
        if prob >= 0.55:
            signal = 'BUY'
        elif prob <= 0.45:
            signal = 'SELL'
        else:
            signal = "HOLD"
            
        return {
            "signal": signal,
            "prob": prob
        }
