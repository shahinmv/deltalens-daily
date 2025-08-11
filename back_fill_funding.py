import sqlite3
import requests
from datetime import datetime, timedelta, timezone
import time
import logging
from requests.exceptions import RequestException
from sqlite3 import Error as SQLiteError
import os

# DB_PATH = os.getenv('DB_PATH', '/app/db.sqlite3')
DB_PATH = '../db.sqlite3'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('startup_logs/funding_rate_collection.log'),
        logging.StreamHandler()
    ]
)

class FundingRateError(Exception):
    """Custom exception for funding rate related errors"""
    pass

def get_latest_funding_time():
    """Get the latest funding time from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT funding_time FROM funding_rates 
            ORDER BY funding_time DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        
        if result:
            latest_time = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            logging.info(f"Found latest funding time in database: {latest_time}")
            return latest_time
        logging.info("No funding times found in database")
        return None
    except SQLiteError as e:
        logging.error(f"Database error while getting latest funding time: {e}")
        raise FundingRateError(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while getting latest funding time: {e}")
        raise FundingRateError(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def get_funding_rate_at_time(timestamp):
    """Get funding rate for a specific timestamp from Binance API"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        "symbol": "BTCUSDT",
        "startTime": int(timestamp.timestamp() * 1000),
        "limit": 1
    }
    
    try:
        logging.debug(f"Fetching funding rate for timestamp: {timestamp}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            funding_rate = float(data[0]['fundingRate'])
            funding_time = datetime.fromtimestamp(data[0]['fundingTime']/1000, timezone.utc)
            
            logging.info(f"Retrieved funding rate {funding_rate} for time {funding_time}")
            return {
                'symbol': 'BTCUSDT',
                'funding_time': funding_time.strftime('%Y-%m-%d %H:%M:%S'),
                'funding_rate': funding_rate
            }
        logging.warning(f"No funding rate data found for timestamp: {timestamp}")
        return None
    except RequestException as e:
        logging.error(f"API request failed: {e}")
        raise FundingRateError(f"API request failed: {e}")
    except (KeyError, ValueError) as e:
        logging.error(f"Error parsing API response: {e}")
        raise FundingRateError(f"Error parsing API response: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while fetching funding rate: {e}")
        raise FundingRateError(f"Unexpected error: {e}")

def store_funding_rate(data):
    """Store funding rate data in the database"""
    if not data:
        logging.warning("Attempted to store empty funding rate data")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if this funding time already exists
        cursor.execute('''
            SELECT 1 FROM funding_rates 
            WHERE funding_time = ?
        ''', (data['funding_time'],))
        
        if cursor.fetchone() is None:
            # Insert the funding rate data
            cursor.execute('''
                INSERT INTO funding_rates (symbol, funding_time, funding_rate)
                VALUES (?, ?, ?)
            ''', (data['symbol'], data['funding_time'], data['funding_rate']))
            
            # Commit the transaction
            conn.commit()
            logging.info(f"Successfully stored funding rate for {data['symbol']} at {data['funding_time']} UTC")
            return True
        else:
            logging.info(f"Funding rate for {data['funding_time']} already exists")
            return False
        
    except SQLiteError as e:
        logging.error(f"Database error while storing funding rate: {e}")
        raise FundingRateError(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while storing funding rate: {e}")
        raise FundingRateError(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main function to collect and store funding rates"""
    try:
        logging.info("Starting funding rate collection process")
        
        # Get the latest funding time from database
        latest_time = get_latest_funding_time()
        
        if latest_time:
            logging.info(f"Latest funding time in database: {latest_time}")
            # Start from the next funding time (8 hours after latest)
            current_time = latest_time + timedelta(hours=8)
        else:
            logging.info("No funding times found in database, starting from 30 days ago")
            current_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Get current time
        end_time = datetime.now(timezone.utc)
        logging.info(f"Collecting funding rates from {current_time} to {end_time}")
        
        # Fetch and store funding rates for each 8-hour interval
        while current_time < end_time:
            try:
                funding_data = get_funding_rate_at_time(current_time)
                if funding_data:
                    store_funding_rate(funding_data)
                current_time += timedelta(hours=8)
                time.sleep(0.1)  # Small delay to avoid rate limiting
            except FundingRateError as e:
                logging.error(f"Error processing funding rate for {current_time}: {e}")
                current_time += timedelta(hours=8)  # Skip to next interval
                continue
        
        logging.info("Funding rate collection process completed successfully")
        
    except Exception as e:
        logging.error(f"Critical error in main process: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program terminated due to error: {e}")
        raise
