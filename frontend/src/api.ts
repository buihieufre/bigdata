import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  prob: number;
  ema_fast: number | null;
  ema_slow: number | null;
}

export interface ChartResponse {
  data: CandleData[];
  thresholds: {
    buy: number;
    sell: number;
  };
  ema_periods: {
    fast: number;
    slow: number;
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
