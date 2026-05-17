import React, { useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import type { ChartResponse, CandleData } from './api';

interface TradingChartProps {
  chartResponse: ChartResponse;
  realtimeCandle?: CandleData | null;
}

export const TradingChart: React.FC<TradingChartProps> = ({ chartResponse, realtimeCandle }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const probChartContainerRef = useRef<HTMLDivElement>(null);
  
  const chartRef = useRef<IChartApi | null>(null);
  const probChartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const data = chartResponse.data;
  const thresholds = chartResponse.thresholds;
  const emaPeriods = chartResponse.ema_periods;

  useEffect(() => {
    if (!chartContainerRef.current || !probChartContainerRef.current) return;

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
      if (probChartRef.current && probChartContainerRef.current) {
        probChartRef.current.applyOptions({ width: probChartContainerRef.current.clientWidth });
      }
    };

    // 1. Create Main Chart
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
        mode: 1, // Normal crosshair
        vertLine: {
          width: 1,
          color: '#758696',
          style: 3, // Dashed
          labelBackgroundColor: '#758696',
        },
        horzLine: {
          width: 1,
          color: '#758696',
          style: 3,
          labelBackgroundColor: '#758696',
        },
      },
      timeScale: {
        borderColor: '#2a2e39',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#2a2e39',
      },
      autoSize: true,
    });
    chartRef.current = chart;

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });
    candlestickSeriesRef.current = candlestickSeries;

    const emaFastSeries = chart.addLineSeries({
      color: '#2962ff',
      lineWidth: 1,
      title: `EMA ${emaPeriods.fast}`,
    });

    const emaSlowSeries = chart.addLineSeries({
      color: '#ff9800',
      lineWidth: 1,
      title: `EMA ${emaPeriods.slow}`,
    });

    // 2. Create Probability Chart
    const probChart = createChart(probChartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2a2e39' },
        horzLines: { color: '#2a2e39' },
      },
      timeScale: {
        borderColor: '#2a2e39',
        timeVisible: true,
        visible: false, // Hide time scale on top chart if syncing
      },
      rightPriceScale: {
        borderColor: '#2a2e39',
        autoScale: false,
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      autoSize: true,
    });
    
    // Set fixed scale for probability chart [0, 1]
    probChart.priceScale('right').applyOptions({
      autoScale: false,
      scaleMargins: {
        top: 0.1,
        bottom: 0.1,
      },
    });

    probChartRef.current = probChart;

    const probSeries = probChart.addAreaSeries({
      topColor: 'rgba(41, 98, 255, 0.4)',
      bottomColor: 'rgba(41, 98, 255, 0.0)',
      lineColor: '#2962ff',
      lineWidth: 2,
      title: 'AI Prob',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });
    
    // Add dynamic Threshold lines
    const buyLine = probChart.addLineSeries({
      color: '#26a69a',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    
    const sellLine = probChart.addLineSeries({
      color: '#ef5350',
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });

    // Sync charts
    const syncCrosshair = (chart1: IChartApi, chart2: IChartApi) => {
      chart1.subscribeCrosshairMove(param => {
        if (!param.time || param.point === undefined || param.point.x < 0 || param.point.y < 0) {
          chart2.clearCrosshairPosition();
        } else {
          // Find equivalent point on the other chart
          chart2.setCrosshairPosition(param.price || 0, param.time, probSeries);
        }
      });
    };
    
    // Better syncing using logical range
    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (range) {
        probChart.timeScale().setVisibleLogicalRange(range);
      }
    });
    probChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (range) {
        chart.timeScale().setVisibleLogicalRange(range);
      }
    });

    syncCrosshair(chart, probChart);
    syncCrosshair(probChart, chart);

    // Prepare data
    const formattedCandles = data.map(d => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
    
    const emaFastData = data.filter(d => d.ema_fast !== null).map(d => ({
      time: d.time as Time,
      value: d.ema_fast as number,
    }));
    
    const emaSlowData = data.filter(d => d.ema_slow !== null).map(d => ({
      time: d.time as Time,
      value: d.ema_slow as number,
    }));

    const probData = data.map(d => ({
      time: d.time as Time,
      value: d.prob,
    }));
    
    const buyLineData = data.map(d => ({ time: d.time as Time, value: thresholds.buy }));
    const sellLineData = data.map(d => ({ time: d.time as Time, value: thresholds.sell }));

    candlestickSeries.setData(formattedCandles);
    emaFastSeries.setData(emaFastData);
    emaSlowSeries.setData(emaSlowData);
    probSeries.setData(probData);
    buyLine.setData(buyLineData);
    sellLine.setData(sellLineData);

    // Set markers for BUY / SELL
    const markers: any[] = [];
    data.forEach(d => {
      if (d.signal === 'BUY') {
        markers.push({
          time: d.time as Time,
          position: 'belowBar',
          color: '#26a69a',
          shape: 'arrowUp',
          text: 'BUY',
        });
      } else if (d.signal === 'SELL') {
        markers.push({
          time: d.time as Time,
          position: 'aboveBar',
          color: '#ef5350',
          shape: 'arrowDown',
          text: 'SELL',
        });
      }
    });
    
    candlestickSeries.setMarkers(markers);

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      probChart.remove();
    };
  }, [data, thresholds.buy, thresholds.sell, emaPeriods.fast, emaPeriods.slow]);

  useEffect(() => {
    if (realtimeCandle && candlestickSeriesRef.current && data.length > 0) {
      const lastDataTime = data[data.length - 1].time;
      // Only update if the realtime candle is at or after the last historical candle
      if (realtimeCandle.time >= lastDataTime) {
        try {
          candlestickSeriesRef.current.update({
            time: realtimeCandle.time as Time,
            open: realtimeCandle.open,
            high: realtimeCandle.high,
            low: realtimeCandle.low,
            close: realtimeCandle.close,
          });
        } catch (e) {
          // Silently ignore update errors during timeframe transitions
          console.warn('Chart update skipped:', e);
        }
      }
    }
  }, [realtimeCandle]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
      <div 
        ref={chartContainerRef} 
        style={{ flex: '1', minHeight: 0, borderBottom: '1px solid #2a2e39' }} 
      />
      <div 
        ref={probChartContainerRef} 
        style={{ height: '25%', minHeight: '150px' }} 
      />
    </div>
  );
};
