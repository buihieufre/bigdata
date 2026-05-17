import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import { fetchChartData } from './api';
import type { ChartResponse } from './api';
import { TradingChart } from './TradingChart';

function App() {
  const [chartResponse, setChartResponse] = useState<ChartResponse | null>(null);
  const [timeframe, setTimeframe] = useState<string>('5min');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [realtimeCandle, setRealtimeCandle] = useState<any>(null);
  const [countdown, setCountdown] = useState<string>('');

  // Use refs to avoid stale closures in WebSocket and timer callbacks
  const closeTimeRef = useRef<number>(0);
  const timeframeRef = useRef<string>(timeframe);
  timeframeRef.current = timeframe;

  const loadData = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true);
    setError(null);
    try {
      const response = await fetchChartData(timeframeRef.current);
      setChartResponse(response);
    } catch (err) {
      setError('Failed to fetch data from backend.');
      console.error(err);
    } finally {
      if (showLoading) setLoading(false);
    }
  }, []);

  // Load data when timeframe changes
  useEffect(() => {
    setRealtimeCandle(null);
    setCountdown('');
    closeTimeRef.current = 0;
    loadData();
    const interval = setInterval(() => loadData(false), 60000);
    return () => clearInterval(interval);
  }, [timeframe, loadData]);

  // Smooth countdown timer - ticks every second independently of WebSocket
  useEffect(() => {
    const timer = setInterval(() => {
      const ct = closeTimeRef.current;
      if (ct <= 0) return;
      const diff = ct - Date.now();
      if (diff <= 0) {
        setCountdown('00:00');
        return;
      }
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      if (h > 0) {
        setCountdown(`${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`);
      } else {
        setCountdown(`${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // WebSocket for Realtime Candle data (fast updates)
  useEffect(() => {
    let alive = true;

    const tfMap: Record<string, string> = {
      '1min': '1m', '5min': '5m', '15min': '15m',
      '1H': '1h', '4H': '4h', '1D': '1d'
    };
    const wsFormat = tfMap[timeframe] || '5m';
    const ws = new WebSocket(`wss://stream.binance.com:9443/ws/btcusdt@kline_${wsFormat}`);

    ws.onmessage = (event) => {
      if (!alive) return;
      const kline = JSON.parse(event.data).k;

      setRealtimeCandle({
        time: Math.floor(kline.t / 1000),
        open: parseFloat(kline.o),
        high: parseFloat(kline.h),
        low: parseFloat(kline.l),
        close: parseFloat(kline.c),
        volume: parseFloat(kline.v),
      });

      // Store close time for the smooth timer
      closeTimeRef.current = kline.T;

      // When candle closes, reload data to get fresh AI predictions
      if (kline.x) {
        loadData(false);
      }
    };

    return () => {
      alive = false;
      ws.close();
    };
  }, [timeframe, loadData]);

  const data = chartResponse?.data || [];
  const latestData = data.length > 0 ? data[data.length - 1] : null;
  const displayData = realtimeCandle ? { ...latestData, ...realtimeCandle } : latestData;
  const thresholds = chartResponse?.thresholds || { buy: 0.65, sell: 0.30 };
  const emaPeriods = chartResponse?.ema_periods || { fast: 9, slow: 21 };

  const priceColor = displayData
    ? (displayData.close >= displayData.open ? '#26a69a' : '#ef5350')
    : '#d1d4dc';

  return (
    <>
      <header className="header">
        <div className="header-title">
          <Activity size={24} color="#2962ff" />
          Real-time Crypto Advisor
        </div>
        <div className="header-controls">
          <div className="live-indicator">
            <div className="pulse"></div>
            LIVE
          </div>
          <select 
            value={timeframe} 
            onChange={(e) => setTimeframe(e.target.value)}
          >
            <option value="1min">1 Minute</option>
            <option value="5min">5 Minutes</option>
            <option value="15min">15 Minutes</option>
            <option value="1H">1 Hour</option>
            <option value="4H">4 Hours</option>
            <option value="1D">1 Day</option>
          </select>
        </div>
      </header>

      <div className="dashboard-layout">
        <div className="panel chart-container">
          {loading ? (
            <div className="loading-container">
              <RefreshCw className="spinner" size={32} />
              <p>Loading market data...</p>
            </div>
          ) : error ? (
            <div className="loading-container" style={{ color: '#ef5350' }}>
              <p>{error}</p>
              <button onClick={() => loadData()} style={{
                marginTop: '16px', padding: '8px 16px', 
                background: '#2962ff', color: 'white', 
                border: 'none', borderRadius: '4px', cursor: 'pointer'
              }}>Retry</button>
            </div>
          ) : (
            chartResponse && <TradingChart chartResponse={chartResponse} realtimeCandle={realtimeCandle} />
          )}
        </div>

        <div className="sidebar">
          <div className="panel card">
            <div className="card-title">AI Prediction Signal</div>
            {displayData ? (
              <>
                <div className={`signal-value signal-${displayData.signal}`}>
                  {displayData.signal}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '14px' }}>
                  <span style={{ color: '#787b86' }}>Probability</span>
                  <span style={{ fontWeight: 600 }}>{(displayData.prob * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-bar-container">
                  <div 
                    className="prob-bar" 
                    style={{ 
                      width: `${displayData.prob * 100}%`,
                      backgroundColor: displayData.prob >= thresholds.buy ? '#26a69a' : displayData.prob <= thresholds.sell ? '#ef5350' : '#2962ff'
                    }}
                  ></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '12px', color: '#787b86' }}>
                  <span>SELL &le; {thresholds.sell.toFixed(2)}</span>
                  <span>BUY &ge; {thresholds.buy.toFixed(2)}</span>
                </div>
              </>
            ) : (
              <p style={{ color: '#787b86' }}>Waiting for data...</p>
            )}
          </div>
          
          <div className="panel card" style={{ flex: 1 }}>
            <div className="card-title">Market Overview (BTC/USDT)</div>
            {displayData && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '14px' }}>
                {/* Live Price + Countdown on the same line */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#787b86' }}>Price</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontWeight: 700, fontSize: '16px', color: priceColor, transition: 'color 0.3s' }}>
                      ${displayData.close.toFixed(2)}
                    </span>
                    {countdown && (
                      <span style={{
                        fontSize: '11px',
                        fontFamily: 'monospace',
                        color: '#ff9800',
                        background: 'rgba(255,152,0,0.12)',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        letterSpacing: '0.5px',
                      }}>
                        ⏱ {countdown}
                      </span>
                    )}
                  </div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#787b86' }}>Volume</span>
                  <span style={{ fontWeight: 600 }}>{displayData.volume.toFixed(2)} BTC</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#787b86' }}>EMA {emaPeriods.fast}</span>
                  <span style={{ fontWeight: 600 }}>{displayData.ema_fast ? `$${displayData.ema_fast.toFixed(2)}` : 'N/A'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#787b86' }}>EMA {emaPeriods.slow}</span>
                  <span style={{ fontWeight: 600 }}>{displayData.ema_slow ? `$${displayData.ema_slow.toFixed(2)}` : 'N/A'}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
