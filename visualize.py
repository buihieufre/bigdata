import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dashboard():
    logger.info("Loading data and model...")
    
    # Load recent data
    try:
        df = pd.read_csv('data/btc_features.csv')
        # Take the last 200 candles for a clean visualization
        df = df.tail(200).reset_index(drop=True)
    except FileNotFoundError:
        logger.error("Data file not found. Please run feature_engineering.py first.")
        return

    try:
        model = joblib.load('models/xgb_model.pkl')
    except FileNotFoundError:
        logger.error("Model not found. Please run train_model.py first.")
        return

    features = [
        'rsi', 'ema9', 'ema21', 'macd', 'macd_signal', 
        'bb_high', 'bb_low', 'return_1', 'return_5', 'volatility', 'volume'
    ]

    # Predict
    logger.info("Generating predictions for visualization...")
    probs = model.predict_proba(df[features])[:, 1]
    df['prob'] = probs
    
    df['signal'] = 'HOLD'
    df.loc[df['prob'] > 0.7, 'signal'] = 'BUY'
    df.loc[df['prob'] < 0.3, 'signal'] = 'SELL'

    # Create figure
    logger.info("Building interactive chart...")
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        subplot_titles=('BTC/USDT Price & Signals', 'XGBoost BUY Probability'),
        row_heights=[0.7, 0.3]
    )

    # 1. Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], 
        name='BTC/USDT'
    ), row=1, col=1)

    # Add BUY signals
    buy_signals = df[df['signal'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['timestamp'], 
            y=buy_signals['low'] * 0.999,
            mode='markers', 
            marker=dict(symbol='triangle-up', color='rgba(0, 255, 0, 0.9)', size=14, line=dict(color='darkgreen', width=1)),
            name='BUY Signal'
        ), row=1, col=1)

    # Add SELL signals
    sell_signals = df[df['signal'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['timestamp'], 
            y=sell_signals['high'] * 1.001,
            mode='markers', 
            marker=dict(symbol='triangle-down', color='rgba(255, 0, 0, 0.9)', size=14, line=dict(color='darkred', width=1)),
            name='SELL Signal'
        ), row=1, col=1)

    # 2. Probability chart
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['prob'], 
        name='Probability', 
        line=dict(color='royalblue', width=2),
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.1)'
    ), row=2, col=1)
    
    # Threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="BUY Threshold", row=2, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="SELL Threshold", row=2, col=1)

    # Update layout
    fig.update_layout(
        title="Real-time Crypto Trading Advisor Dashboard",
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    # Save to HTML
    os.makedirs('output', exist_ok=True)
    output_path = 'output/dashboard.html'
    fig.write_html(output_path)
    logger.info(f"Dashboard successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
