import psycopg2
import requests
from datetime import datetime, timedelta, timezone
import time
import logging
from requests.exceptions import RequestException
from psycopg2 import Error as PostgreSQLError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection string
POSTGRES_URL = os.getenv('POSTGRES_URL', 'postgresql://postgres:AdkiHmmAoHPWhHzphxCwbqcDRvfmRnjJ@ballast.proxy.rlwy.net:49094/railway')

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('startup_logs/funding_rate_collection.log'),
#         logging.StreamHandler()
#     ]
# )

class FundingRateError(Exception):
    """Custom exception for funding rate related errors"""
    pass

def get_latest_funding_time():
    """Get the latest funding time from the database"""
    print("[DEBUG] Starting get_latest_funding_time()")
    print(f"[DEBUG] Using PostgreSQL URL: {POSTGRES_URL[:50]}...")
    
    try:
        print("[DEBUG] Attempting to connect to PostgreSQL...")
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        print("[DEBUG] Successfully connected to PostgreSQL")
        
        print("[DEBUG] Executing query to get latest funding time...")
        cursor.execute('''
            SELECT funding_time FROM funding_rates 
            ORDER BY funding_time DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        print(f"[DEBUG] Query result: {result}")
        
        if result:
            latest_time = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            print(f"[DEBUG] Parsed latest funding time: {latest_time}")
            logging.info(f"Found latest funding time in database: {latest_time}")
            return latest_time
        print("[DEBUG] No results found - database appears to be empty")
        logging.info("No funding times found in database")
        return None
    except PostgreSQLError as e:
        print(f"[DEBUG] PostgreSQL error: {e}")
        logging.error(f"Database error while getting latest funding time: {e}")
        raise FundingRateError(f"Database error: {e}")
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {e}")
        logging.error(f"Unexpected error while getting latest funding time: {e}")
        raise FundingRateError(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals():
            print("[DEBUG] Closing database connection")
            conn.close()

def get_funding_rate_at_time(timestamp):
    """Get funding rate for a specific timestamp from Binance API"""
    print(f"[DEBUG] Fetching funding rate for timestamp: {timestamp}")
    
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        "symbol": "BTCUSDT",
        "startTime": int(timestamp.timestamp() * 1000),
        "limit": 1
    }
    print(f"[DEBUG] API URL: {url}")
    print(f"[DEBUG] API params: {params}")
    
    try:
        print(f"[DEBUG] Making API request to Binance...")
        logging.debug(f"Fetching funding rate for timestamp: {timestamp}")
        response = requests.get(url, params=params)
        print(f"[DEBUG] API response status code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"[DEBUG] API response data: {data}")
        
        if data and len(data) > 0:
            funding_rate = float(data[0]['fundingRate'])
            funding_time = datetime.fromtimestamp(data[0]['fundingTime']/1000, timezone.utc)
            print(f"[DEBUG] Parsed funding rate: {funding_rate}, funding time: {funding_time}")
            
            logging.info(f"Retrieved funding rate {funding_rate} for time {funding_time}")
            return {
                'symbol': 'BTCUSDT',
                'funding_time': funding_time.strftime('%Y-%m-%d %H:%M:%S'),
                'funding_rate': funding_rate
            }
        print(f"[DEBUG] No funding rate data found in API response")
        logging.warning(f"No funding rate data found for timestamp: {timestamp}")
        return None
    except RequestException as e:
        print(f"[DEBUG] API request exception: {e}")
        logging.error(f"API request failed: {e}")
        raise FundingRateError(f"API request failed: {e}")
    except (KeyError, ValueError) as e:
        print(f"[DEBUG] Data parsing error: {e}")
        logging.error(f"Error parsing API response: {e}")
        raise FundingRateError(f"Error parsing API response: {e}")
    except Exception as e:
        print(f"[DEBUG] Unexpected error in API call: {e}")
        logging.error(f"Unexpected error while fetching funding rate: {e}")
        raise FundingRateError(f"Unexpected error: {e}")

def store_funding_rate(data):
    """Store funding rate data in the database"""
    print(f"[DEBUG] Attempting to store funding rate data: {data}")
    
    if not data:
        print("[DEBUG] No data provided - returning False")
        logging.warning("Attempted to store empty funding rate data")
        return False
    
    try:
        print("[DEBUG] Connecting to PostgreSQL for data storage...")
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        print("[DEBUG] Successfully connected to PostgreSQL for storage")
        
        # Create table if it doesn't exist
        print("[DEBUG] Creating funding_rates table if it doesn't exist...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS funding_rates (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                funding_time VARCHAR(19) NOT NULL,
                funding_rate DECIMAL(20,10) NOT NULL,
                UNIQUE(symbol, funding_time)
            )
        ''')
        print("[DEBUG] Table creation/verification completed")
        
        # Check if this funding time already exists
        print(f"[DEBUG] Checking if funding time {data['funding_time']} already exists...")
        cursor.execute('''
            SELECT 1 FROM funding_rates 
            WHERE funding_time = %s
        ''', (data['funding_time'],))
        
        existing = cursor.fetchone()
        print(f"[DEBUG] Existing record check result: {existing}")
        
        if existing is None:
            print(f"[DEBUG] No existing record found - inserting new data...")
            # Insert the funding rate data
            cursor.execute('''
                INSERT INTO funding_rates (symbol, funding_time, funding_rate)
                VALUES (%s, %s, %s)
            ''', (data['symbol'], data['funding_time'], data['funding_rate']))
            
            print(f"[DEBUG] Rows affected by insert: {cursor.rowcount}")
            
            # Commit the transaction
            conn.commit()
            print(f"[DEBUG] Transaction committed successfully")
            logging.info(f"Successfully stored funding rate for {data['symbol']} at {data['funding_time']} UTC")
            return True
        else:
            print(f"[DEBUG] Record already exists - skipping insert")
            logging.info(f"Funding rate for {data['funding_time']} already exists")
            return False
        
    except PostgreSQLError as e:
        print(f"[DEBUG] PostgreSQL error during storage: {e}")
        logging.error(f"Database error while storing funding rate: {e}")
        raise FundingRateError(f"Database error: {e}")
    except Exception as e:
        print(f"[DEBUG] Unexpected error during storage: {e}")
        logging.error(f"Unexpected error while storing funding rate: {e}")
        raise FundingRateError(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals():
            print("[DEBUG] Closing storage database connection")
            conn.close()

def main():
    """Main function to collect and store funding rates"""
    print("[DEBUG] " + "="*60)
    print("[DEBUG] Starting funding rate collection main process")
    print("[DEBUG] " + "="*60)
    
    try:
        logging.info("Starting funding rate collection process")
        
        # Get the latest funding time from database
        print("[DEBUG] Step 1: Getting latest funding time from database...")
        latest_time = get_latest_funding_time()
        print(f"[DEBUG] Latest funding time result: {latest_time}")
        
        if latest_time:
            logging.info(f"Latest funding time in database: {latest_time}")
            # Start from the next funding time (8 hours after latest)
            current_time = latest_time + timedelta(hours=8)
            print(f"[DEBUG] Starting from next interval: {current_time}")
        else:
            logging.info("No funding times found in database, starting from 30 days ago")
            current_time = datetime.now(timezone.utc) - timedelta(days=30)
            print(f"[DEBUG] Starting from 30 days ago: {current_time}")
        
        # Get current time
        end_time = datetime.now(timezone.utc)
        print(f"[DEBUG] End time: {end_time}")
        logging.info(f"Collecting funding rates from {current_time} to {end_time}")
        
        # Calculate total intervals
        total_intervals = int((end_time - current_time).total_seconds() / (8 * 3600))
        print(f"[DEBUG] Total 8-hour intervals to process: {total_intervals}")
        
        # Fetch and store funding rates for each 8-hour interval
        interval_count = 0
        successful_inserts = 0
        skipped_existing = 0
        
        while current_time < end_time:
            interval_count += 1
            print(f"[DEBUG] Processing interval {interval_count}/{total_intervals}: {current_time}")
            
            try:
                funding_data = get_funding_rate_at_time(current_time)
                if funding_data:
                    result = store_funding_rate(funding_data)
                    if result:
                        successful_inserts += 1
                        print(f"[DEBUG] ✓ Successfully inserted data for {current_time}")
                    else:
                        skipped_existing += 1
                        print(f"[DEBUG] - Skipped existing data for {current_time}")
                else:
                    print(f"[DEBUG] ⚠ No funding data returned from API for {current_time}")
                    
                current_time += timedelta(hours=8)
                time.sleep(0.1)  # Small delay to avoid rate limiting
                
            except FundingRateError as e:
                print(f"[DEBUG] ❌ FundingRateError for {current_time}: {e}")
                logging.error(f"Error processing funding rate for {current_time}: {e}")
                current_time += timedelta(hours=8)  # Skip to next interval
                continue
        
        print(f"[DEBUG] " + "="*60)
        print(f"[DEBUG] Collection completed successfully!")
        print(f"[DEBUG] Total intervals processed: {interval_count}")
        print(f"[DEBUG] Successful inserts: {successful_inserts}")
        print(f"[DEBUG] Skipped existing: {skipped_existing}")
        print(f"[DEBUG] " + "="*60)
        
        logging.info("Funding rate collection process completed successfully")
        
    except Exception as e:
        print(f"[DEBUG] ❌ Critical error in main process: {e}")
        logging.error(f"Critical error in main process: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program terminated due to error: {e}")
        raise
