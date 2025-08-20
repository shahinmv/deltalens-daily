import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection details
POSTGRES_URL = os.getenv('POSTGRES_URL', 'postgresql://postgres:password@localhost:5432/database')

# Table names
BTC_DAILY = 'btc_daily_ohlcv'
OI_TABLE_NAME = 'open_interest'
FD_TABLE_NAME = 'funding_rates'
NEWS_TABLE_NAME = 'news'

def load_btc_ohlcv():
    engine = create_engine(POSTGRES_URL)
    query = f"SELECT * FROM {BTC_DAILY}"
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    # df.index = df.index.date
    return df

def load_open_interest():
    engine = create_engine(POSTGRES_URL)
    query = f"SELECT * FROM {OI_TABLE_NAME}"
    df = pd.read_sql_query(query, engine, parse_dates=['timestamp'])
    engine.dispose()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    daily_oi = df.resample('1D').mean()
    return daily_oi

def load_funding_rates():
    engine = create_engine(POSTGRES_URL)
    query = f"SELECT * FROM {FD_TABLE_NAME}"
    df = pd.read_sql_query(query, engine, parse_dates=['funding_time'])
    engine.dispose()
    df['funding_time'] = pd.to_datetime(df['funding_time'])
    df.set_index('funding_time', inplace=True)
    daily_funding_rate = df.drop(columns=['symbol']).resample('1D').last()
    return daily_funding_rate

def load_news():
    engine = create_engine(POSTGRES_URL)
    query = f"SELECT * FROM {NEWS_TABLE_NAME}"
    df = pd.read_sql_query(query, engine, parse_dates=['date'])
    engine.dispose()
    return df

def load_all_data(months=None):
    """
    Load all data, optionally filtering to the last N months from the latest available data
    """
    btc_ohlcv = load_btc_ohlcv()
    daily_oi = load_open_interest()
    daily_funding_rate = load_funding_rates()
    df_news = load_news()
    
    if months is not None:
        # Find the latest date across all datasets
        latest_dates = []
        if not btc_ohlcv.empty:
            latest_dates.append(btc_ohlcv.index.max())
        # if not daily_oi.empty:
        #     latest_dates.append(daily_oi.index.max())
        # if not daily_funding_rate.empty:
        #     latest_dates.append(daily_funding_rate.index.max())
        # if not df_news.empty:
        #     latest_dates.append(pd.to_datetime(df_news['date']).max())
        
        if latest_dates:
            latest_date = max(latest_dates)
            cutoff_date = latest_date - timedelta(days=months * 30)
            
            # Filter each dataset
            btc_ohlcv = btc_ohlcv[btc_ohlcv.index >= cutoff_date]
            daily_oi = daily_oi[daily_oi.index >= cutoff_date]
            daily_funding_rate = daily_funding_rate[daily_funding_rate.index >= cutoff_date]
            df_news = df_news[pd.to_datetime(df_news['date']) >= cutoff_date]
            
            print(f"Latest data date: {latest_date.strftime('%Y-%m-%d')}")
            print(f"Filtered data to last {months} months (from {cutoff_date.strftime('%Y-%m-%d')})")
    
    return btc_ohlcv, daily_oi, daily_funding_rate, df_news
