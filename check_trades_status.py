import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('trade_status_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeStatusChecker:
    def __init__(self, db_connection_string='postgresql://postgres:AdkiHmmAoHPWhHzphxCwbqcDRvfmRnjJ@ballast.proxy.rlwy.net:49094/railway'):
        self.db_connection_string = db_connection_string
        self.create_trade_status_table()
    
    def create_trade_status_table(self):
        """Create the trade_status table if it doesn't exist"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_status (
                    id SERIAL PRIMARY KEY,
                    signal_id INTEGER NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
                    entry_date DATE,
                    current_price NUMERIC,
                    hit_target BOOLEAN DEFAULT FALSE,
                    hit_stop_loss BOOLEAN DEFAULT FALSE,
                    is_expired BOOLEAN DEFAULT FALSE,
                    pnl_percentage NUMERIC,
                    exit_reason TEXT,
                    exit_price NUMERIC,
                    exit_date DATE,
                    exit_timestamp TIMESTAMP,
                    days_active INTEGER DEFAULT 0,
                    last_checked TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES iterative_trading_signals (id),
                    UNIQUE(signal_id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Trade status table created/verified successfully")
            
        except Exception as e:
            logger.error(f"Error creating trade status table: {e}")
            raise

    def get_ohlcv_data(self, start_date, end_date=None):
        """Get OHLCV data from btc_daily_ohlcv table for the specified date range"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            query = """
                SELECT datetime, open, high, low, close, volume 
                FROM btc_daily_ohlcv 
                WHERE datetime >= %s AND datetime <= %s
                ORDER BY datetime ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()
            
            if df.empty:
                return None
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None

    def get_latest_ohlcv_date(self):
        """Get the latest date available in btc_daily_ohlcv table"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            query = "SELECT MAX(datetime) FROM btc_daily_ohlcv"
            cursor.execute(query)
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return datetime.strptime(result[0], '%Y-%m-%d').date()
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest OHLCV date: {e}")
            return None

    def get_active_signals(self):
        """Get all signals that need status checking"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT its.id, its.prediction_date, its.symbol, its.signal_type, 
                       its.entry_price, its.target_price, its.stop_loss, 
                       its.expires_at, its.status as signal_status,
                       ts.status as trade_status, ts.id as trade_status_id
                FROM benchmark_trading_signals its
                LEFT JOIN trade_status ts ON its.id = ts.signal_id
                WHERE (ts.status IS NULL 
                       OR ts.status IN ('ACTIVE', 'ACTIVE_PENDING_DATA'))
                  AND its.status = 'ACTIVE'
                ORDER BY its.prediction_date DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            signals = []
            for row in rows:
                signals.append({
                    'id': row[0],
                    'prediction_date': row[1],
                    'symbol': row[2],
                    'signal_type': row[3],
                    'entry_price': row[4],
                    'target_price': row[5],
                    'stop_loss': row[6],
                    'expires_at': row[7],
                    'signal_status': row[8],
                    'trade_status': row[9],
                    'trade_status_id': row[10]
                })
            
            logger.info(f"Found {len(signals)} signals to check")
            return signals
            
        except Exception as e:
            logger.error(f"Error fetching active signals: {e}")
            return []

    def check_expiry(self, expires_at):
        """Check if a signal has expired"""
        try:
            if not expires_at:
                return False
            
            expires_datetime = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            current_datetime = datetime.now(timezone.utc)
            
            return current_datetime > expires_datetime
            
        except Exception as e:
            logger.error(f"Error checking expiry for {expires_at}: {e}")
            return False

    def calculate_pnl(self, signal_type, entry_price, current_price):
        """Calculate PnL percentage"""
        try:
            if signal_type.upper() == 'LONG':
                return ((current_price - entry_price) / entry_price) * 100
            elif signal_type.upper() == 'SHORT':
                return ((entry_price - current_price) / entry_price) * 100
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating PnL: {e}")
            return 0.0

    def parse_prediction_date(self, date_str):
        """Parse prediction date from various formats"""
        try:
            # Try standard date format first
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except:
            try:
                # Try with time component
                return datetime.strptime(date_str[:10], '%Y-%m-%d').date()
            except Exception as e:
                logger.error(f"Could not parse date: {date_str}")
                raise e

    def analyze_trade_with_ohlcv(self, signal, ohlcv_data):
        """Analyze trade using OHLCV data to check for target/stop loss hits"""
        signal_type = signal['signal_type'].upper()
        entry_price = signal['entry_price']
        target_price = signal['target_price']
        stop_loss = signal['stop_loss']
        
        prediction_date = self.parse_prediction_date(signal['prediction_date'])
        
        hit_target = False
        hit_stop_loss = False
        exit_date = None
        exit_price = None
        days_active = 0
        
        for date, row in ohlcv_data.iterrows():
            current_date = date.date()
            days_from_prediction = (current_date - prediction_date).days
            
            if days_from_prediction < 0:
                continue
                
            days_active = days_from_prediction + 1
            
            if signal_type == 'LONG':
                if row['high'] >= target_price:
                    hit_target = True
                    exit_date = current_date.strftime('%Y-%m-%d')
                    exit_price = target_price
                    break
                elif row['low'] <= stop_loss:
                    hit_stop_loss = True
                    exit_date = current_date.strftime('%Y-%m-%d')
                    exit_price = stop_loss
                    break
                    
            elif signal_type == 'SHORT':
                if row['low'] <= target_price:
                    hit_target = True
                    exit_date = current_date.strftime('%Y-%m-%d')
                    exit_price = target_price
                    break
                elif row['high'] >= stop_loss:
                    hit_stop_loss = True
                    exit_date = current_date.strftime('%Y-%m-%d')
                    exit_price = stop_loss
                    break
        
        # If no target/stop hit, use last available price
        if not hit_target and not hit_stop_loss and not ohlcv_data.empty:
            exit_price = ohlcv_data.iloc[-1]['close']
        
        return hit_target, hit_stop_loss, exit_date, exit_price, days_active

    def update_trade_status(self, signal, exit_price, hit_target, hit_stop_loss, is_expired, exit_date, days_active, has_ohlcv_data):
        """Update or insert trade status"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            prediction_date = self.parse_prediction_date(signal['prediction_date'])
            entry_date = prediction_date.strftime('%Y-%m-%d')
            
            # Calculate PnL if we have a price
            pnl_percentage = 0.0
            if exit_price:
                pnl_percentage = self.calculate_pnl(signal['signal_type'], signal['entry_price'], exit_price)
            
            # Determine status and exit reason
            if hit_target:
                status = 'TARGET_HIT'
                exit_reason = 'Target reached'
                exit_timestamp = datetime.now().isoformat()
            elif hit_stop_loss:
                status = 'STOP_LOSS_HIT'
                exit_reason = 'Stop loss triggered'
                exit_timestamp = datetime.now().isoformat()
            elif is_expired:
                status = 'EXPIRED'
                exit_reason = 'Signal expired'
                exit_timestamp = datetime.now().isoformat()
            elif not has_ohlcv_data:
                status = 'ACTIVE_PENDING_DATA'
                exit_reason = 'Awaiting OHLCV data'
                exit_timestamp = None
            else:
                status = 'ACTIVE'
                exit_reason = 'Trade in progress'
                exit_timestamp = None
            
            current_time = datetime.now().isoformat()
            
            # Update or insert record
            if signal['trade_status_id']:
                cursor.execute("""
                    UPDATE trade_status 
                    SET status = %s, entry_date = %s, current_price = %s, hit_target = %s, hit_stop_loss = %s, 
                        is_expired = %s, pnl_percentage = %s, exit_reason = %s, 
                        exit_price = %s, exit_date = %s, exit_timestamp = %s, 
                        days_active = %s, last_checked = %s, updated_at = %s
                    WHERE signal_id = %s
                """, (status, entry_date, exit_price, hit_target, hit_stop_loss, is_expired, 
                      pnl_percentage, exit_reason, exit_price, exit_date, exit_timestamp,
                      days_active, current_time, current_time, signal['id']))
            else:
                cursor.execute("""
                    INSERT INTO trade_status 
                    (signal_id, status, entry_date, current_price, hit_target, hit_stop_loss, 
                     is_expired, pnl_percentage, exit_reason, exit_price, exit_date,
                     exit_timestamp, days_active, last_checked)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (signal['id'], status, entry_date, exit_price, hit_target, hit_stop_loss, 
                      is_expired, pnl_percentage, exit_reason, exit_price, exit_date,
                      exit_timestamp, days_active, current_time))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated signal {signal['id']}: {status} "
                       f"(Entry: {entry_date}, Price: ${exit_price or 0:.2f}, PnL: {pnl_percentage:.2f}%, Days: {days_active})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating trade status for signal {signal['id']}: {e}")
            return False

    def check_all_trades(self):
        """Main function to check all active trades"""
        logger.info("Starting trade status check...")
        
        latest_data_date = self.get_latest_ohlcv_date()
        if not latest_data_date:
            logger.error("Could not determine latest OHLCV data date")
            return
        
        logger.info(f"Latest OHLCV data available: {latest_data_date}")
        
        signals = self.get_active_signals()
        if not signals:
            logger.info("No active signals to check")
            return
        
        logger.info(f"Processing {len(signals)} signals")
        
        checked_count = 0
        updated_count = 0
        
        for signal in signals:
            try:
                logger.info(f"Processing signal {signal['id']}: {signal['signal_type']} on {signal['prediction_date']}")
                
                prediction_date = self.parse_prediction_date(signal['prediction_date'])
                is_expired = self.check_expiry(signal['expires_at'])
                
                # Get OHLCV data
                start_date = prediction_date.strftime('%Y-%m-%d')
                end_date = None
                
                if is_expired:
                    try:
                        expires_date = datetime.fromisoformat(signal['expires_at'].replace('Z', '+00:00')).date()
                        end_date = expires_date.strftime('%Y-%m-%d')
                    except:
                        pass
                
                ohlcv_data = self.get_ohlcv_data(start_date, end_date)
                has_ohlcv_data = ohlcv_data is not None and not ohlcv_data.empty
                
                # Initialize default values
                hit_target = False
                hit_stop_loss = False
                exit_date = None
                exit_price = None
                days_active = 0
                
                if has_ohlcv_data:
                    logger.info(f"Found {len(ohlcv_data)} days of OHLCV data for signal {signal['id']}")
                    hit_target, hit_stop_loss, exit_date, exit_price, days_active = self.analyze_trade_with_ohlcv(signal, ohlcv_data)
                else:
                    logger.warning(f"No OHLCV data available for signal {signal['id']} from {start_date}")
                    # Calculate days since prediction for pending trades
                    days_active = (datetime.now().date() - prediction_date).days
                
                # Update status regardless of OHLCV data availability
                if self.update_trade_status(signal, exit_price, hit_target, hit_stop_loss, is_expired, exit_date, days_active, has_ohlcv_data):
                    updated_count += 1
                
                checked_count += 1
                
            except Exception as e:
                logger.error(f"Error processing signal {signal['id']}: {e}")
                continue
        
        logger.info(f"Trade status check completed. Checked: {checked_count}, Updated: {updated_count}")

    def get_trade_status_summary(self):
        """Get a summary of all trade statuses"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ts.status, COUNT(*) as count, 
                       AVG(ts.pnl_percentage) as avg_pnl,
                       MIN(ts.pnl_percentage) as min_pnl,
                       MAX(ts.pnl_percentage) as max_pnl,
                       AVG(ts.days_active) as avg_days_active
                FROM trade_status ts
                GROUP BY ts.status
                ORDER BY count DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            summary = {}
            for row in rows:
                summary[row[0]] = {
                    'count': row[1],
                    'avg_pnl': row[2] if row[2] else 0,
                    'min_pnl': row[3] if row[3] else 0,
                    'max_pnl': row[4] if row[4] else 0,
                    'avg_days_active': row[5] if row[5] else 0
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting trade status summary: {e}")
            return {}

def main():
    """Main execution function"""
    try:
        checker = TradeStatusChecker()
        checker.check_all_trades()
        
        # Print summary
        summary = checker.get_trade_status_summary()
        if summary:
            logger.info("=== TRADE STATUS SUMMARY ===")
            for status, data in summary.items():
                logger.info(f"{status}: {data['count']} trades, "
                           f"Avg PnL: {data['avg_pnl']:.2f}% "
                           f"(Range: {data['min_pnl']:.2f}% to {data['max_pnl']:.2f}%), "
                           f"Avg Days Active: {data['avg_days_active']:.1f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()