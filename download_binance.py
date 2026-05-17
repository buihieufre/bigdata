import requests
import pandas as pd
import datetime
import time
import os
import logging
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
LIMIT = 1000
OUTPUT_DIR = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'btc_raw.parquet')

def get_historical_klines(symbol, interval, start_ts, end_ts):
    """
    Get Historical Klines from Binance
    """
    url = "https://api.binance.com/api/v3/klines"
    
    klines = []
    current_start = start_ts
    
    total_expected = (end_ts - start_ts) / (60 * 1000)
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': LIMIT,
            'startTime': current_start,
            'endTime': end_ts
        }
            
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                logger.warning("Rate limit hit. Sleeping for 10 seconds.")
                time.sleep(10)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            klines.extend(data)
            
            # Update start_ts to the last kline timestamp + 1
            current_start = data[-1][0] + 1
            
            if current_start >= end_ts:
                break
                
            if len(klines) % 100000 == 0:
                logger.info(f"Downloaded {len(klines)} / {total_expected:.0f} candles ({(len(klines)/total_expected*100):.1f}%)")
                
            # Rate limiting - Binance limit is 1200 weight/min, this costs 2 weight.
            # We can safely do 10 requests per second.
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            time.sleep(5) # Retry after 5 sec
            
    return klines

def main():
    logger.info(f"Starting to download {SYMBOL} data (3 Years)...")
    
    # Download last 3 years of data
    end_time = datetime.datetime.now()
    start_time = end_time - relativedelta(years=3)
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    logger.info(f"Downloading data from {start_time} to {end_time}")
    
    klines = get_historical_klines(SYMBOL, INTERVAL, start_ts, end_ts)
    
    if not klines:
        logger.error("No data fetched.")
        return
        
    logger.info("Processing downloaded data...")
    # Process klines
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    # Select important columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to Parquet
    logger.info(f"Saving {len(df)} rows to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Data saved successfully. Final Shape: {df.shape}")

if __name__ == "__main__":
    main()
