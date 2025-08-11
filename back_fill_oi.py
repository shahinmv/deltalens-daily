import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
from binance.client import Client
from datetime import timezone
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('startup_logs/open_interest_collection.log'),
        logging.StreamHandler()
    ]
)

# Initialize Binance client without API keys
client = Client()

# DB_PATH = os.getenv('DB_PATH', '/app/db.sqlite3')
DB_PATH = '../db.sqlite3'

def get_latest_timestamp():
    """Get the latest timestamp from the open_interest table"""
    logging.info("Connecting to database to get latest timestamp...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT MAX(timestamp) FROM open_interest")
        latest_timestamp = cursor.fetchone()[0]
        if latest_timestamp is None:
            logging.info("No existing data found, starting from 30 days ago")
            return int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        logging.info(f"Found latest timestamp in database: {latest_timestamp}")
        # Convert datetime string to Unix timestamp in milliseconds
        dt = datetime.strptime(latest_timestamp, '%Y-%m-%d %H:%M:%S')
        # Add 2 days to the last timestamp
        # dt = dt + timedelta(days=15)
        logging.info(f"Starting from: {dt}")
        return int(dt.timestamp() * 1000)
    finally:
        conn.close()

def fetch_open_interest_data(start_time):
    """Fetch open interest data from Binance"""
    end_time = int(datetime.now().timestamp() * 1000)
    all_data = []
    chunk_count = 0
    last_timestamp = start_time
    
    logging.info(f"Fetching data from {datetime.fromtimestamp(start_time/1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_time/1000, tz=timezone.utc)}")
    
    while start_time < end_time:
        try:
            chunk_count += 1
            
            # Calculate end time for this chunk (500 5-minute intervals after start_time)
            chunk_end_time = start_time + (500 * 5 * 60 * 1000)  # 500 intervals * 5 minutes * 60 seconds * 1000 milliseconds
            
            # Ensure we don't exceed the final end_time
            if chunk_end_time > end_time:
                chunk_end_time = end_time
            
            # Fetch data in chunks of 500 records (Binance's maximum limit)
            klines = client.futures_open_interest_hist(
                symbol='BTCUSDT',
                period='5m',
                startTime=start_time,
                endTime=chunk_end_time,
                limit=500  # Use maximum allowed limit
            )
            
            if not klines:
                logging.info("No more data available")
                break
                
            # Check if we're getting the same data repeatedly
            if klines[-1]['timestamp'] <= last_timestamp:
                logging.info("Reached duplicate data, stopping")
                break
                
            all_data.extend(klines)
            last_timestamp = klines[-1]['timestamp']
            start_time = klines[-1]['timestamp'] + 300000  # Add 5 minutes in milliseconds
            
            logging.info(f"Fetched chunk {chunk_count}, total records: {len(all_data)}")
            logging.info(f"Next chunk will start from: {datetime.fromtimestamp(start_time/1000, tz=timezone.utc)}")
            
            # Respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            time.sleep(1)
            continue
    
    logging.info(f"Total records fetched: {len(all_data)}")
    return all_data

def insert_data_to_db(data):
    """Insert fetched data into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        inserted_count = 0
        for item in data:
            # Convert Unix timestamp to UTC datetime string
            timestamp_str = datetime.fromtimestamp(item['timestamp']/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("""
                INSERT OR IGNORE INTO open_interest 
                (timestamp, close_settlement, close_quote)
                VALUES (?, ?, ?)
            """, (
                timestamp_str,
                float(item['sumOpenInterest']),
                float(item['sumOpenInterestValue'])
            ))
            inserted_count += cursor.rowcount
        
        conn.commit()
        return inserted_count
    finally:
        conn.close()

def main():
    # Get the latest timestamp from the database
    start_time = get_latest_timestamp()
    # logging.info(f"Starting data fetch from: {datetime.fromtimestamp(start_time/1000, tz=timezone.utc)}")
    
    # Fetch new data
    new_data = fetch_open_interest_data(start_time)
    
    if new_data:
        # Insert new data into the database
        inserted_count = insert_data_to_db(new_data)
        if inserted_count > 0:
            logging.info(f"Successfully appended {inserted_count} new rows")
        else:
            logging.info("No new rows appended (all data already exists)")
    else:
        logging.info("No new data to append")

if __name__ == "__main__":
    main()
