import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import { fetchChartData } from './api';
import type { ChartResponse } from './api';
import { TradingChart } from './TradingChart';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

function App() {
  const [chartResponse, setChartResponse] = useState<ChartResponse | null>(null);
  const [timeframe, setTimeframe] = useState<string>('5min');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [realtimeCandle, setRealtimeCandle] = useState<any>(null);
  const [countdown, setCountdown] = useState<string>('');
  const [tableLimit, setTableLimit] = useState<number>(10);

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
  const marketCtx = chartResponse?.market_context;

  // Process data for the rolling table
  const recentTableData = data.slice(-tableLimit).map((d, index, arr) => {
    // If it's the last element and we have a realtime update, merge them
    if (index === arr.length - 1 && realtimeCandle) {
      return { ...d, ...realtimeCandle };
    }
    return d;
  }).reverse(); // Reverse to show newest on top

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
        <PanelGroup direction="vertical">
          <Panel defaultSize={70} minSize={30}>
            <PanelGroup direction="horizontal">
              <Panel defaultSize={75} minSize={30}>
                <div className="panel chart-container" style={{ height: '100%', border: 'none', background: 'var(--panel-bg)', borderRadius: '8px', overflow: 'hidden' }}>
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
              </Panel>
              
              <PanelResizeHandle className="resize-handle-vertical" />
              
              <Panel defaultSize={25} minSize={15}>
                <div className="sidebar" style={{ height: '100%' }}>
                  <div className="panel card" style={{ height: '100%', overflowY: 'auto' }}>
                    <div className="card-title">ICT Signal</div>
                    {displayData ? (
                      <>
                        {/* ICT Bias badge */}
                        <div className={`signal-value signal-${displayData.signal}`}>
                          {displayData.bias ?? displayData.signal}
                        </div>

                        {/* Probability Up/Down */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '14px' }}>
                          <span style={{ color: '#787b86' }}>Prob Up</span>
                          <span style={{ fontWeight: 600, color: '#26a69a' }}>
                            {((displayData.probability_up ?? displayData.prob) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                          <span style={{ color: '#787b86' }}>Prob Down</span>
                          <span style={{ fontWeight: 600, color: '#ef5350' }}>
                            {((displayData.probability_down ?? (1 - displayData.prob)) * 100).toFixed(1)}%
                          </span>
                        </div>

                        {/* Probability bar */}
                        <div className="prob-bar-container" style={{ marginTop: '8px' }}>
                          <div
                            className="prob-bar"
                            style={{
                              width: `${(displayData.probability_up ?? displayData.prob) * 100}%`,
                              backgroundColor: displayData.signal === 'BUY' ? '#26a69a' : displayData.signal === 'SELL' ? '#ef5350' : '#2962ff'
                            }}
                          ></div>
                        </div>


                        {/* Killzone / Macro */}
                        {marketCtx && (
                          <div style={{ marginTop: '10px', padding: '6px 8px', borderRadius: '4px', background: marketCtx.macro_active ? 'rgba(41,98,255,0.15)' : 'rgba(120,123,134,0.1)', fontSize: '12px' }}>
                            <div style={{ color: marketCtx.macro_active ? '#2962ff' : '#787b86', fontWeight: 600 }}>
                              {marketCtx.killzone}
                            </div>
                            <div style={{ color: '#787b86', marginTop: '2px' }}>
                              HTF: <span style={{ color: marketCtx.htf_bias === 'BULLISH' ? '#26a69a' : marketCtx.htf_bias === 'BEARISH' ? '#ef5350' : '#787b86', fontWeight: 600 }}>{marketCtx.htf_bias}</span>
                            </div>
                          </div>
                        )}

                        {/* ICT Context */}
                        {displayData.ict_context && (
                          <div style={{ marginTop: '8px', fontSize: '11px', color: '#787b86', lineHeight: 1.4, wordBreak: 'break-word' }}>
                            {displayData.ict_context}
                          </div>
                        )}
                      </>
                    ) : (
                      <p style={{ color: '#787b86' }}>Waiting for data...</p>
                    )}
                  </div>
                </div>
              </Panel>
            </PanelGroup>
          </Panel>

          <PanelResizeHandle className="resize-handle-horizontal" />

          <Panel defaultSize={30} minSize={15}>
            <div className="panel card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div className="card-title" style={{ margin: 0 }}>Real-time ICT Features</div>
                <div>
                  <span style={{ fontSize: '12px', color: '#787b86' }}>Show:</span>
                  <select 
                    className="limit-selector" 
                    value={tableLimit} 
                    onChange={(e) => setTableLimit(Number(e.target.value))}
                  >
                    <option value={5}>5 Rows</option>
                    <option value={10}>10 Rows</option>
                    <option value={15}>15 Rows</option>
                    <option value={20}>20 Rows</option>
                  </select>
                </div>
              </div>

              <div className="ict-table-container" style={{ flex: 1, overflowY: 'auto' }}>
                <table className="ict-table">
                  <thead>
                    <tr>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Time</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Price</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>HTF Trend</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>MSS</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Liq Sweep</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Bias</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Prob (U/D)</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Macro</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>BISI</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>SIBI</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Context</th>
                      <th style={{ position: 'sticky', top: 0, backgroundColor: 'var(--panel-bg)', zIndex: 1 }}>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentTableData.map((row: any, i: number) => {
                      const date = new Date((row.time || 0) * 1000);
                      const timeStr = date.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
                      
                      let htfLabel = "RANGING";
                      let htfColor = "#787b86";
                      if (row.htf_trend === 1) { htfLabel = "BULLISH"; htfColor = "#26a69a"; }
                      if (row.htf_trend === -1) { htfLabel = "BEARISH"; htfColor = "#ef5350"; }
                      
                      let mssLabel = "-";
                      let mssColor = "#787b86";
                      if (row.mss_strength && Math.abs(row.mss_strength) > 0.1) {
                        mssLabel = row.mss_strength > 0 ? "BULL MSS" : "BEAR MSS";
                        mssColor = row.mss_strength > 0 ? "#26a69a" : "#ef5350";
                      }

                      let liqLabel = "-";
                      let liqColor = "#787b86";
                      if (row.liquidity_sweep_detected === 1) { liqLabel = "SWEPT LOW"; liqColor = "#26a69a"; }
                      if (row.liquidity_sweep_detected === 2) { liqLabel = "SWEPT HIGH"; liqColor = "#ef5350"; }

                      let sigLabel = (row.signal || "NEUTRAL").replace('_', ' ');
                      let sigColor = "#787b86";
                      if (sigLabel.includes("LONG")) sigColor = "#26a69a";
                      if (sigLabel.includes("SHORT")) sigColor = "#ef5350";

                      let biasLabel = row.bias || "-";
                      let biasColor = "#787b86";
                      if (biasLabel === "LONG") biasColor = "#26a69a";
                      if (biasLabel === "SHORT") biasColor = "#ef5350";
                      
                      let probUp = row.probability_up ?? row.prob ?? 0;
                      let probDn = row.probability_down ?? (1 - probUp);
                      let probLabel = `${(probUp*100).toFixed(0)}% / ${(probDn*100).toFixed(0)}%`;
                      
                      let macroLabel = "-";
                      if (row.macro_window === 1) macroLabel = "London";
                      if (row.macro_window === 2) macroLabel = "NY";
                      
                      let bisiLabel = row.inside_bisi_mid_zone === 1 ? "IN" : "-";
                      let sibiLabel = row.inside_sibi_mid_zone === 1 ? "IN" : "-";
                      
                      let ctxLabel = row.ict_context || "-";
                      if (ctxLabel.length > 30) ctxLabel = ctxLabel.substring(0, 30) + "...";

                      return (
                        <tr key={`${row.time}-${i}`}>
                          <td style={{ color: '#787b86' }}>
                            {timeStr}
                            {i === 0 && countdown && (
                              <span style={{ marginLeft: '6px', fontSize: '10px', color: '#ff9800' }}>
                                ({countdown})
                              </span>
                            )}
                          </td>
                          <td style={{ fontWeight: 600 }}>${Number(row.close).toFixed(1)}</td>
                          <td style={{ color: htfColor }}>{htfLabel}</td>
                          <td style={{ color: mssColor }}>{mssLabel}</td>
                          <td style={{ color: liqColor }}>{liqLabel}</td>
                          <td style={{ color: biasColor }}>{biasLabel}</td>
                          <td style={{ color: '#787b86', fontSize: '12px' }}>{probLabel}</td>
                          <td style={{ color: macroLabel !== '-' ? '#2962ff' : '#787b86' }}>{macroLabel}</td>
                          <td style={{ color: bisiLabel === 'IN' ? '#26a69a' : '#787b86' }}>{bisiLabel}</td>
                          <td style={{ color: sibiLabel === 'IN' ? '#ef5350' : '#787b86' }}>{sibiLabel}</td>
                          <td style={{ color: '#787b86', fontSize: '12px' }}>{ctxLabel}</td>
                          <td style={{ color: sigColor, fontWeight: 600 }}>{sigLabel}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </Panel>
        </PanelGroup>
      </div>
    </>
  );
}

export default App;
