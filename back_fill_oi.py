import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import time
from binance.client import Client
from datetime import timezone
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('startup_logs/open_interest_collection.log'),
#         logging.StreamHandler()
#     ]
# )

# Initialize Binance client without API keys
client = Client()

# PostgreSQL connection string
POSTGRES_URL = os.getenv('POSTGRES_URL', 'postgresql://postgres:AdkiHmmAoHPWhHzphxCwbqcDRvfmRnjJ@ballast.proxy.rlwy.net:49094/railway')

def get_latest_timestamp():
    """Get the latest timestamp from the open_interest table"""
    print("[DEBUG] Starting get_latest_timestamp()")
    print(f"[DEBUG] Using PostgreSQL URL: {POSTGRES_URL[:50]}...")
    
    logging.info("Connecting to database to get latest timestamp...")
    
    try:
        print("[DEBUG] Attempting to connect to PostgreSQL...")
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        print("[DEBUG] Successfully connected to PostgreSQL")
        
        print("[DEBUG] Executing query to get latest timestamp...")
        cursor.execute("SELECT MAX(timestamp) FROM open_interest")
        latest_timestamp = cursor.fetchone()[0]
        print(f"[DEBUG] Query result: {latest_timestamp}")
        
        if latest_timestamp is None:
            start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            start_date = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
            print(f"[DEBUG] No existing data found - starting from 30 days ago: {start_date}")
            logging.info("No existing data found, starting from 30 days ago")
            return start_time
            
        logging.info(f"Found latest timestamp in database: {latest_timestamp}")
        print(f"[DEBUG] Found latest timestamp string: {latest_timestamp}")
        
        # Convert datetime string to Unix timestamp in milliseconds
        dt = datetime.strptime(latest_timestamp, '%Y-%m-%d %H:%M:%S')
        timestamp_ms = int(dt.timestamp() * 1000)
        
        print(f"[DEBUG] Parsed datetime: {dt}")
        print(f"[DEBUG] Converted to timestamp (ms): {timestamp_ms}")
        
        logging.info(f"Starting from: {dt}")
        return timestamp_ms
        
    except Exception as e:
        print(f"[DEBUG] Error in get_latest_timestamp: {e}")
        raise
    finally:
        if 'conn' in locals():
            print("[DEBUG] Closing database connection")
            conn.close()

def fetch_open_interest_data(start_time):
    """Fetch open interest data from Binance"""
    print(f"[DEBUG] Starting fetch_open_interest_data with start_time: {start_time}")
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_date = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
    end_date = datetime.fromtimestamp(end_time/1000, tz=timezone.utc)
    
    print(f"[DEBUG] Start date: {start_date}")
    print(f"[DEBUG] End date: {end_date}")
    print(f"[DEBUG] Time range (ms): {start_time} to {end_time}")
    
    # Calculate expected data points
    total_minutes = (end_time - start_time) / (1000 * 60)
    expected_records = int(total_minutes / 5)  # 5-minute intervals
    print(f"[DEBUG] Expected approximate records: {expected_records}")
    
    all_data = []
    chunk_count = 0
    last_timestamp = start_time
    
    logging.info(f"Fetching data from {start_date} to {end_date}")
    
    while start_time < end_time:
        try:
            chunk_count += 1
            print(f"[DEBUG] Processing chunk {chunk_count}")
            
            # Calculate end time for this chunk (500 5-minute intervals after start_time)
            chunk_end_time = start_time + (500 * 5 * 60 * 1000)  # 500 intervals * 5 minutes * 60 seconds * 1000 milliseconds
            
            # Ensure we don't exceed the final end_time
            if chunk_end_time > end_time:
                chunk_end_time = end_time
                
            chunk_start_date = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
            chunk_end_date = datetime.fromtimestamp(chunk_end_time/1000, tz=timezone.utc)
            print(f"[DEBUG] Chunk {chunk_count} range: {chunk_start_date} to {chunk_end_date}")
            
            # Fetch data in chunks of 500 records (Binance's maximum limit)
            print(f"[DEBUG] Making Binance API call for chunk {chunk_count}...")
            klines = client.futures_open_interest_hist(
                symbol='BTCUSDT',
                period='5m',
                startTime=start_time,
                endTime=chunk_end_time,
                limit=500  # Use maximum allowed limit
            )
            
            print(f"[DEBUG] API returned {len(klines) if klines else 0} records for chunk {chunk_count}")
            
            if not klines:
                print(f"[DEBUG] No more data available from API - stopping")
                logging.info("No more data available")
                break
                
            # Check if we're getting the same data repeatedly
            if klines[-1]['timestamp'] <= last_timestamp:
                print(f"[DEBUG] Reached duplicate data (last_timestamp: {last_timestamp}, current: {klines[-1]['timestamp']}) - stopping")
                logging.info("Reached duplicate data, stopping")
                break
                
            all_data.extend(klines)
            last_timestamp = klines[-1]['timestamp']
            start_time = klines[-1]['timestamp'] + 300000  # Add 5 minutes in milliseconds
            
            next_start_date = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
            print(f"[DEBUG] Chunk {chunk_count} completed: {len(klines)} records, total so far: {len(all_data)}")
            print(f"[DEBUG] Next chunk will start from: {next_start_date}")
            
            logging.info(f"Fetched chunk {chunk_count}, total records: {len(all_data)}")
            logging.info(f"Next chunk will start from: {next_start_date}")
            
            # Respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"[DEBUG] Error in chunk {chunk_count}: {e}")
            logging.error(f"Error fetching data: {e}")
            time.sleep(1)
            continue
    
    print(f"[DEBUG] Data fetching completed. Total records: {len(all_data)}")
    if all_data:
        first_date = datetime.fromtimestamp(all_data[0]['timestamp']/1000, tz=timezone.utc)
        last_date = datetime.fromtimestamp(all_data[-1]['timestamp']/1000, tz=timezone.utc)
        print(f"[DEBUG] Data range: {first_date} to {last_date}")
    
    logging.info(f"Total records fetched: {len(all_data)}")
    return all_data

def insert_data_to_db(data):
    """Insert fetched data into the database"""
    print(f"[DEBUG] Starting database insertion with {len(data)} records")
    
    if not data:
        print("[DEBUG] No data to insert - returning 0")
        return 0
    
    try:
        print("[DEBUG] Connecting to PostgreSQL for data insertion...")
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        print("[DEBUG] Successfully connected for data insertion")
        
        # Create table if it doesn't exist
        print("[DEBUG] Creating open_interest table if it doesn't exist...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(19) NOT NULL,
                close_settlement DECIMAL(20,2) NOT NULL,
                close_quote DECIMAL(20,2) NOT NULL,
                UNIQUE(timestamp)
            )
        """)
        print("[DEBUG] Table creation/verification completed")
        
        inserted_count = 0
        duplicate_count = 0
        error_count = 0
        
        print(f"[DEBUG] Processing {len(data)} records for insertion...")
        
        for i, item in enumerate(data, 1):
            try:
                # Convert Unix timestamp to UTC datetime string
                timestamp_str = datetime.fromtimestamp(item['timestamp']/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                if i <= 3 or i % 100 == 0 or i == len(data):  # Debug first 3, every 100th, and last
                    print(f"[DEBUG] Processing record {i}/{len(data)}: {timestamp_str}")
                    print(f"[DEBUG]   - Open Interest: {item['sumOpenInterest']}")
                    print(f"[DEBUG]   - Open Interest Value: {item['sumOpenInterestValue']}")
                
                cursor.execute("""
                    INSERT INTO open_interest 
                    (timestamp, close_settlement, close_quote)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING
                """, (
                    timestamp_str,
                    float(item['sumOpenInterest']),
                    float(item['sumOpenInterestValue'])
                ))
                
                if cursor.rowcount > 0:
                    inserted_count += 1
                else:
                    duplicate_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"[DEBUG] Error processing record {i}: {e}")
                continue
        
        print(f"[DEBUG] Committing transaction...")
        conn.commit()
        print(f"[DEBUG] Transaction committed successfully")
        
        print(f"[DEBUG] Insertion summary:")
        print(f"[DEBUG]   - Total records processed: {len(data)}")
        print(f"[DEBUG]   - Successfully inserted: {inserted_count}")
        print(f"[DEBUG]   - Duplicates skipped: {duplicate_count}")
        print(f"[DEBUG]   - Errors encountered: {error_count}")
        
        return inserted_count
        
    except Exception as e:
        print(f"[DEBUG] Database insertion error: {e}")
        raise
    finally:
        if 'conn' in locals():
            print("[DEBUG] Closing database connection")
            conn.close()

def main():
    print("[DEBUG] " + "="*60)
    print("[DEBUG] Starting open interest data collection main process")
    print("[DEBUG] " + "="*60)
    
    try:
        # Get the latest timestamp from the database
        print("[DEBUG] Step 1: Getting latest timestamp from database...")
        start_time = get_latest_timestamp()
        start_date = datetime.fromtimestamp(start_time/1000, tz=timezone.utc)
        print(f"[DEBUG] Will start data collection from: {start_date}")
        
        # Fetch new data
        print("[DEBUG] Step 2: Fetching open interest data from Binance...")
        new_data = fetch_open_interest_data(start_time)
        
        if new_data:
            print(f"[DEBUG] Step 3: Inserting {len(new_data)} records into database...")
            # Insert new data into the database
            inserted_count = insert_data_to_db(new_data)
            
            if inserted_count > 0:
                print(f"[DEBUG] ✓ Successfully inserted {inserted_count} new records")
                logging.info(f"Successfully appended {inserted_count} new rows")
            else:
                print(f"[DEBUG] ⚠ No new records inserted (all data already exists)")
                logging.info("No new rows appended (all data already exists)")
                
            # Show data range summary
            if new_data:
                first_date = datetime.fromtimestamp(new_data[0]['timestamp']/1000, tz=timezone.utc)
                last_date = datetime.fromtimestamp(new_data[-1]['timestamp']/1000, tz=timezone.utc)
                print(f"[DEBUG] Data range processed: {first_date} to {last_date}")
                
        else:
            print(f"[DEBUG] ⚠ No new data fetched from Binance")
            logging.info("No new data to append")
            
        print("[DEBUG] " + "="*60)
        print("[DEBUG] Open interest data collection completed successfully!")
        print("[DEBUG] " + "="*60)
        
    except Exception as e:
        print(f"[DEBUG] ✗ Critical error in main process: {e}")
        logging.error(f"Critical error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
