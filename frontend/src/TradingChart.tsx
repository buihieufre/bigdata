import React, { useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import type { ChartResponse, CandleData } from './api';

interface TradingChartProps {
  chartResponse: ChartResponse;
  realtimeCandle?: CandleData | null;
}

export const TradingChart: React.FC<TradingChartProps> = ({ chartResponse, realtimeCandle }) => {
  const chartContainerRef     = useRef<HTMLDivElement>(null);
  const probChartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef              = useRef<IChartApi | null>(null);
  const probChartRef          = useRef<IChartApi | null>(null);
  const candleSeriesRef       = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const { data, thresholds } = chartResponse;

  useEffect(() => {
    if (!chartContainerRef.current || !probChartContainerRef.current) return;

    const handleResize = () => {
      chartRef.current?.applyOptions({ width: chartContainerRef.current!.clientWidth });
      probChartRef.current?.applyOptions({ width: probChartContainerRef.current!.clientWidth });
    };

    // ── Main chart ──────────────────────────────────────────────────────────
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2a2e39' },
        horzLines: { color: '#2a2e39' },
      },
      crosshair: {
        mode: 1,
        vertLine: { width: 1, color: '#758696', style: 3, labelBackgroundColor: '#758696' },
        horzLine: { width: 1, color: '#758696', style: 3, labelBackgroundColor: '#758696' },
      },
      localization: {
        timeFormatter: (time: Time) => {
          const date = new Date((time as number) * 1000);
          return date.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false, year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
        }
      },
      timeScale: { 
        borderColor: '#2a2e39', 
        timeVisible: true, 
        secondsVisible: false,
        tickMarkFormatter: (time: Time) => {
          const date = new Date((time as number) * 1000);
          return date.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit' });
        }
      },
      rightPriceScale: { borderColor: '#2a2e39' },
      autoSize: true,
    });
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a', downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    candleSeriesRef.current = candleSeries;

    // ── Probability chart ───────────────────────────────────────────────────
    const probChart = createChart(probChartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2a2e39' },
        horzLines: { color: '#2a2e39' },
      },
      localization: {
        timeFormatter: (time: Time) => {
          const date = new Date((time as number) * 1000);
          return date.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false, year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
        }
      },
      timeScale: { borderColor: '#2a2e39', timeVisible: true, visible: false },
      rightPriceScale: {
        borderColor: '#2a2e39', autoScale: false,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      autoSize: true,
    });
    probChartRef.current = probChart;

    const probSeries = probChart.addAreaSeries({
      topColor: 'rgba(41,98,255,0.4)', bottomColor: 'rgba(41,98,255,0.0)',
      lineColor: '#2962ff', lineWidth: 2, title: 'AI Prob',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });

    const buyLine = probChart.addLineSeries({
      color: '#26a69a', lineWidth: 1, lineStyle: 2,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
    });
    const sellLine = probChart.addLineSeries({
      color: '#ef5350', lineWidth: 1, lineStyle: 2,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
    });

    // ── Sync charts ─────────────────────────────────────────────────────────
    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (range) probChart.timeScale().setVisibleLogicalRange(range);
    });
    probChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (range) chart.timeScale().setVisibleLogicalRange(range);
    });

    // ── Data ────────────────────────────────────────────────────────────────
    candleSeries.setData(data.map(d => ({
      time: d.time as Time, open: d.open, high: d.high, low: d.low, close: d.close,
    })));

    probSeries.setData(data.map(d => ({ time: d.time as Time, value: d.prob })));
    buyLine.setData(data.map(d => ({ time: d.time as Time, value: thresholds.buy })));
    sellLine.setData(data.map(d => ({ time: d.time as Time, value: thresholds.sell })));

    // ── LONG / SHORT markers only ───────────────────────────────────────────
    const markers: any[] = [];
    for (const d of data) {
      if (d.signal === 'LONG' || d.signal === 'STRONG_LONG') {
        markers.push({
          time: d.time,
          position: 'belowBar',
          color: '#26a69a',
          shape: 'arrowUp',
          text: d.signal.replace('_', ' '),
        });
      } else if (d.signal === 'SHORT' || d.signal === 'STRONG_SHORT') {
        markers.push({
          time: d.time,
          position: 'aboveBar',
          color: '#ef5350',
          shape: 'arrowDown',
          text: d.signal.replace('_', ' '),
        });
      }
    }
    candleSeries.setMarkers(markers);

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      probChart.remove();
    };
  }, [data, thresholds.buy, thresholds.sell]);

  // Realtime candle update
  useEffect(() => {
    if (realtimeCandle && candleSeriesRef.current && data.length > 0) {
      const lastTime = data[data.length - 1].time;
      if (realtimeCandle.time >= lastTime) {
        try {
          candleSeriesRef.current.update({
            time: realtimeCandle.time as Time,
            open: realtimeCandle.open, high: realtimeCandle.high,
            low: realtimeCandle.low,   close: realtimeCandle.close,
          });
        } catch (e) {
          console.warn('Chart update skipped:', e);
        }
      }
    }
  }, [realtimeCandle]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
      <div ref={chartContainerRef}
        style={{ flex: '1', minHeight: 0, borderBottom: '1px solid #2a2e39' }} />
      <div ref={probChartContainerRef}
        style={{ height: '25%', minHeight: '150px' }} />
    </div>
  );
};
