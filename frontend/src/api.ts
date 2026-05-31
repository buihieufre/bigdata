import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal: 'LONG' | 'SHORT' | 'NEUTRAL' | 'STRONG_LONG' | 'STRONG_SHORT';
  prob: number;
  ema_fast: number | null;
  ema_slow: number | null;
  // ICT fields
  bias?: 'LONG' | 'SHORT' | 'NEUTRAL';
  probability_up?: number;
  probability_down?: number;
  ict_context?: string;
  macro_window?: number;  // 0=none, 1=London, 2=NY
  htf_trend?: number;    // -1=bear, 0=ranging, 1=bull
  mss_strength?: number;
  liquidity_sweep_detected?: number;
  inside_bisi_mid_zone?: number;
  inside_sibi_mid_zone?: number;
}

export interface FvgZone {
  type: 'bull' | 'bear';
  filled: boolean;
  time: number;
  fvg_high: number;
  fvg_low: number;
  fvg_mid: number;
  fill_ratio: number;
}

export interface MssEvent {
  time: number;
  direction: 'bull' | 'bear';
  strength: number;
}

export interface CisdEvent {
  time: number;
  state: 1 | -1;
}

export interface LiqEvent {
  time: number;
  type: 1 | 2; // 1=sweep_low (bullish), 2=sweep_high (bearish)
}

export interface Overlays {
  fvg_zones:   FvgZone[];
  mss_events:  MssEvent[];
  cisd_events: CisdEvent[];
  liq_events:  LiqEvent[];
}

export interface ChartResponse {
  data: CandleData[];
  thresholds: { buy: number; sell: number };
  ema_periods: { fast: number; slow: number };
  overlays?: Overlays;
  market_context?: {
    htf_bias: string;
    killzone: string;
    macro_active: boolean;
  };
}

export const fetchChartData = async (timeframe: string = '1min'): Promise<ChartResponse> => {
  try {
    const response = await axios.get<ChartResponse>(`${API_BASE_URL}/data?timeframe=${timeframe}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};
