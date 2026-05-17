import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model and base data
logger.info("Loading model and data...")
model = joblib.load('models/xgb_model.pkl')
df_raw = pd.read_parquet('data/btc_raw.parquet')
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

# Initialize Dash App
app = dash.Dash(__name__, title="TradingView Crypto Advisor")

app.layout = html.Div([
    html.Div([
        html.H2("Real-time Crypto Trading Advisor", style={'color': '#d1d4dc', 'margin': '0', 'padding': '15px'}),
        html.Div([
            html.Label("Timeframe: ", style={'color': '#d1d4dc', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[
                    {'label': '1 Minute (Native)', 'value': '1min'},
                    {'label': '5 Minutes', 'value': '5min'},
                    {'label': '15 Minutes', 'value': '15min'},
                    {'label': '1 Hour', 'value': '1H'},
                    {'label': '4 Hours', 'value': '4H'}
                ],
                value='5min',
                clearable=False,
                style={'width': '200px', 'color': 'black', 'display': 'inline-block', 'verticalAlign': 'middle'}
            )
        ], style={'padding': '0 15px 15px 15px'})
    ], style={'borderBottom': '1px solid #2a2e39', 'marginBottom': '10px'}),
    
    dcc.Loading(
        id="loading-chart",
        type="default",
        color="#2962ff",
        children=dcc.Graph(id='price-chart', style={'height': '85vh'})
    )
], style={'backgroundColor': '#131722', 'minHeight': '100vh', 'fontFamily': 'Trebuchet MS, Roboto, Ubuntu, sans-serif'})

@app.callback(
    Output('price-chart', 'figure'),
    [Input('timeframe-dropdown', 'value')]
)
def update_chart(timeframe):
    logger.info(f"Processing data for timeframe: {timeframe}")
    
    # Resample data
    if timeframe == '1min':
        df = df_raw.copy()
    else:
        df = df_raw.set_index('timestamp').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
    # Generate features
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['volatility'] = df['return_1'].rolling(window=20).std()
    
    df = df.dropna().reset_index(drop=True)
    
    features = ['rsi', 'ema9', 'ema21', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'return_1', 'return_5', 'volatility', 'volume']
    
    # Predict probabilities
    probs = model.predict_proba(df[features])[:, 1]
    df['prob'] = probs
    df['signal'] = 'HOLD'
    
    # Use adaptive threshold for larger timeframes as variance changes
    # For 1m it was 0.7/0.3. For resampled data, it may compress. 
    # But we stick to the rule for simplicity.
    df.loc[df['prob'] >= 0.55, 'signal'] = 'BUY'
    df.loc[df['prob'] <= 0.45, 'signal'] = 'SELL'
    
    # Limit to last 5000 candles to ensure browser performance
    df = df.tail(5000)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.03, row_heights=[0.8, 0.2]
    )
                        
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='BTC/USDT', 
        increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
        decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350'
    ), row=1, col=1)
    
    # BUY signals
    buy_signals = df[df['signal'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['timestamp'], y=buy_signals['low']*0.999,
            mode='markers', marker=dict(symbol='triangle-up', color='#26a69a', size=14, line=dict(color='white', width=1)),
            name='BUY'
        ), row=1, col=1)
                                 
    # SELL signals
    sell_signals = df[df['signal'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['timestamp'], y=sell_signals['high']*1.001,
            mode='markers', marker=dict(symbol='triangle-down', color='#ef5350', size=14, line=dict(color='white', width=1)),
            name='SELL'
        ), row=1, col=1)
                                 
    # Probability Line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['prob'], name='Buy Probability', 
        line=dict(color='#2962ff', width=2), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'
    ), row=2, col=1)
    
    # Thresholds
    fig.add_hline(y=0.7, line_dash="dash", line_color="#26a69a", row=2, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="#ef5350", row=2, col=1)
    
    # Update layout to look like TradingView
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=20, b=50),
        hovermode='x unified',
        showlegend=False,
        xaxis=dict(
            showgrid=True, gridcolor='#2a2e39',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='#2a2e39',
                activecolor='#2962ff',
                font=dict(color='#d1d4dc')
            )
        ),
        xaxis2=dict(showgrid=True, gridcolor='#2a2e39'),
        yaxis=dict(showgrid=True, gridcolor='#2a2e39', side='right', tickformat='.2f'),
        yaxis2=dict(showgrid=True, gridcolor='#2a2e39', side='right', range=[0, 1])
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="AI Probability", row=2, col=1)
    
    return fig

if __name__ == '__main__':
    logger.info("Starting Dash server on port 5000...")
    app.run(debug=False, host='0.0.0.0', port=5000)
    