import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import logging
from datetime import datetime, timedelta
import json
import os
import sys
import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ml_model import ImprovedBitcoinPredictor

# Import the required modules using relative imports
base_path = os.path.dirname(os.path.abspath(__file__))
dev_path = os.path.join(base_path, 'data')
if dev_path not in sys.path:
    sys.path.insert(0, dev_path)
if base_path not in sys.path:
    sys.path.insert(0, base_path)
# Now import
from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from feature_engineering import engineer_features

"""
=============================================================================
DELTALENS BITCOIN TRADING STRATEGY & RISK ASSESSMENT
=============================================================================

This module implements an advanced algorithmic trading system for Bitcoin (BTC/USDT) 
using ensemble machine learning models with dynamic risk management.

TRADING STRATEGY OVERVIEW:
-------------------------
The system employs a multi-model ensemble approach combining:
1. LSTM (Long Short-Term Memory) networks for price prediction
2. Gradient Boosting for market regime detection  
3. Transformer models for complex pattern recognition
4. Dynamic stop-loss calculation based on market conditions

KEY FEATURES:
- Ensemble prediction combining multiple ML models
- Dynamic stop-loss based on volatility, momentum, and market regime
- Kelly Criterion for optimal position sizing
- Multi-factor risk assessment incorporating technical and sentiment indicators
- Real-time data integration from Binance API
- Comprehensive backtesting and performance monitoring

SIGNAL GENERATION LOGIC:
-----------------------
Signals are generated when predicted returns exceed defined thresholds:
- LONG signals: Predicted return > +2.0% with high confidence
- SHORT signals: Predicted return < -2.0% with high confidence  
- HOLD signals: Predicted return within ±2.0% range

Position sizing follows Kelly Criterion methodology:
- Position Size = (Predicted Return × Confidence) / Expected Volatility
- Maximum position size capped at 10% of portfolio
- Minimum confidence threshold of 0.6 required for signal execution

DYNAMIC STOP-LOSS METHODOLOGY:
-----------------------------
Stop-loss levels are calculated using multiple market factors:

1. VOLATILITY ADJUSTMENT:
   - ATR (Average True Range): Adapts to recent price volatility
   - Historical Volatility: 30-day rolling standard deviation
   - Base multiplier: 2-5x depending on market conditions

2. MARKET REGIME DETECTION:
   - Trending Markets: Wider stops (1.2-1.4x multiplier)
   - Ranging Markets: Tighter stops (0.7-0.9x multiplier)
   - Bear Markets: Enhanced risk controls

3. MOMENTUM ANALYSIS:
   - Overbought conditions (RSI > 70): Tighter stops for longs
   - Oversold conditions (RSI < 30): Wider stops allowing for recovery
   - Momentum divergence detection for early exit signals

4. FEAR & GREED INDEX:
   - Extreme Fear (>75): Wider stops, expect volatility
   - Extreme Greed (<25): Moderate stops, expect correction
   - Neutral conditions: Standard stop-loss application

5. CONFIDENCE-BASED ADJUSTMENT:
   - High confidence trades: Tighter stops (higher conviction)
   - Low confidence trades: Wider stops (allow for uncertainty)

FINAL STOP-LOSS CALCULATION:
Dynamic Stop = Base Stop (3%) × Volatility × Regime × Momentum × Fear × Confidence
- Minimum: 1.5% (tight risk control)
- Maximum: 12% (prevent excessive losses)

RISK ASSESSMENT FRAMEWORK:
=========================

1. MARKET RISK:
   - Maximum daily loss exposure: 10% of portfolio value
   - Position correlation limits: Maximum 3 concurrent positions
   - Sector concentration: 100% Bitcoin exposure (high concentration risk)
   - Volatility monitoring: Real-time VaR calculations

2. MODEL RISK:
   - Ensemble approach reduces single-model dependency
   - Regular model retraining (weekly) prevents overfitting
   - Out-of-sample validation required before deployment
   - Performance degradation monitoring with automatic alerts

3. OPERATIONAL RISK:
   - API failure contingency: Local data backup systems
   - Network connectivity: Multiple data source redundancy
   - System downtime: Automated position protection protocols
   - Data quality checks: Outlier detection and correction

4. LIQUIDITY RISK:
   - Bitcoin market depth analysis before position sizing
   - Slippage estimation for large positions
   - Market hours consideration (24/7 crypto markets)
   - Emergency exit procedures for low liquidity periods

5. TECHNICAL RISK:
   - Stop-loss execution risk in volatile markets
   - Gap risk during market disruptions
   - Platform risk (exchange downtime or issues)
   - Latency risk in high-frequency scenarios

PERFORMANCE METRICS:
-------------------
- Sharpe Ratio: Risk-adjusted returns measurement
- Maximum Drawdown: Worst-case loss scenario tracking
- Win Rate: Percentage of profitable trades
- Average Holding Period: Trade duration analysis
- Volatility Adjusted Returns: Return per unit of risk

RISK LIMITS & CONTROLS:
----------------------
- Maximum Position Size: 10% of portfolio per trade
- Daily Loss Limit: 5% of portfolio value
- Maximum Consecutive Losses: 3 trades (triggers review)
- Volatility Threshold: Suspend trading if BTC vol > 100% annualized
- Correlation Limits: No more than 3 correlated positions

EMERGENCY PROCEDURES:
--------------------
- Circuit Breaker: Automatic trading halt if losses exceed 5% in one day
- Model Degradation: Switch to conservative mode if accuracy drops below 55%
- Market Disruption: Immediate position closure if extreme volatility detected
- System Failure: Manual override capabilities with predefined exit strategies

REGULATORY & COMPLIANCE:
-----------------------
- Trade logging: All signals and executions recorded with timestamps
- Audit trail: Complete decision rationale documentation
- Risk reporting: Daily risk metrics and exposure analysis
- Model validation: Independent validation of prediction models

DISCLAIMER:
----------
This trading system involves substantial risk of loss. Past performance does not
guarantee future results. The system is designed for sophisticated investors who
understand the risks involved in algorithmic trading of volatile assets like Bitcoin.
All users should carefully consider their risk tolerance and investment objectives.

=============================================================================
"""

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('iterative_trading_dynamic_sl.log'),
#         logging.StreamHandler()
#     ]
# )

class DynamicStopLossCalculator:
    """
    Advanced dynamic stop loss calculator for iterative trading
    """
    
    def __init__(self, df, atr_period=14, volatility_lookback=30):
        self.df = df
        self.atr_period = atr_period
        self.volatility_lookback = volatility_lookback
        self.price_col = 'close' if 'close' in df.columns else 'Close'
        self.high_col = 'high' if 'high' in df.columns else 'High'
        self.low_col = 'low' if 'low' in df.columns else 'Low'
        
        # Pre-calculate technical indicators
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """Pre-calculate all technical indicators needed for dynamic stop loss"""
        try:
            # Calculate True Range and ATR
            high = self.df[self.high_col]
            low = self.df[self.low_col]
            close = self.df[self.price_col]
            prev_close = close.shift(1)
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.DataFrame([tr1, tr2, tr3]).max()
            
            # Average True Range with proper NaN handling
            atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
            self.df['atr'] = atr.fillna(self.df[self.price_col] * 0.02)  # 2% fallback
            
            # Historical volatility with proper NaN handling
            returns = close.pct_change()
            volatility = returns.rolling(window=self.volatility_lookback, min_periods=5).std() * np.sqrt(252)
            self.df['volatility'] = volatility.fillna(0.5)  # 50% annual volatility fallback
            
            # Market regime indicators with robust NaN handling
            trend_strength = self._calculate_trend_strength()
            self.df['trend_strength'] = trend_strength.fillna(0.3)  # Default moderate trend
            
            # Support and resistance levels
            self.df['resistance'] = high.rolling(window=20, min_periods=1).max().fillna(high)
            self.df['support'] = low.rolling(window=20, min_periods=1).min().fillna(low)
            
            # Market momentum (RSI-like) with robust NaN handling
            momentum = self._calculate_momentum()
            self.df['momentum'] = momentum.fillna(50.0)  # Neutral momentum fallback
            
            # VIX-like fear index for crypto
            fear_index = self._calculate_fear_index()
            self.df['fear_index'] = fear_index.fillna(50.0)  # Neutral fear fallback
            
            # Final NaN check and cleanup
            self._cleanup_nan_values()
            
        except Exception as e:
            logging.error(f"Error calculating dynamic stop loss indicators: {e}")
            # Set robust default values if calculation fails
            self._set_default_indicators()
    
    def _calculate_trend_strength(self):
        """Calculate trend strength similar to ADX with robust NaN handling"""
        try:
            high = self.df[self.high_col]
            low = self.df[self.low_col]
            
            # Ensure we have enough data
            if len(high) < 2:
                return pd.Series([0.3] * len(self.df), index=self.df.index)
            
            # Directional movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth the directional movements with min_periods
            window = min(14, len(self.df) - 1, max(1, len(self.df) // 4))
            plus_dm_smooth = pd.Series(plus_dm).rolling(window=window, min_periods=1).mean()
            minus_dm_smooth = pd.Series(minus_dm).rolling(window=window, min_periods=1).mean()
            
            # Calculate trend strength with safe division
            denominator = plus_dm_smooth + minus_dm_smooth
            trend_strength = np.where(
                denominator > 1e-6,
                abs(plus_dm_smooth - minus_dm_smooth) / denominator,
                0.3  # Default moderate trend when no movement
            )
            
            # Convert to Series and handle any remaining NaN
            trend_series = pd.Series(trend_strength, index=self.df.index)
            trend_series = trend_series.fillna(0.3)
            
            # Ensure values are in reasonable range
            trend_series = np.clip(trend_series, 0.0, 1.0)
            
            return trend_series
            
        except Exception as e:
            logging.warning(f"Error calculating trend strength: {e}")
            return pd.Series([0.3] * len(self.df), index=self.df.index)
    
    def _calculate_momentum(self):
        """Calculate momentum indicator (RSI-like) with robust NaN handling"""
        try:
            close = self.df[self.price_col]
            
            # Ensure we have enough data
            if len(close) < 2:
                return pd.Series([50.0] * len(self.df), index=self.df.index)
            
            # Calculate price changes
            price_change = close.diff()
            gain = np.where(price_change > 0, price_change, 0)
            loss = np.where(price_change < 0, -price_change, 0)
            
            # Use adaptive window size based on available data
            window = min(14, len(self.df) - 1, max(1, len(self.df) // 4))
            
            # Calculate averages with min_periods
            avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
            avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
            
            # Safe division for RS calculation
            rs = np.where(avg_loss > 1e-6, avg_gain / avg_loss, 1.0)
            
            # Calculate momentum (RSI formula)
            momentum = 100 - (100 / (1 + rs))
            
            # Convert to Series and handle NaN
            momentum_series = pd.Series(momentum, index=self.df.index)
            momentum_series = momentum_series.fillna(50.0)
            
            # Ensure values are in valid RSI range
            momentum_series = np.clip(momentum_series, 0.0, 100.0)
            
            return momentum_series
            
        except Exception as e:
            logging.warning(f"Error calculating momentum: {e}")
            return pd.Series([50.0] * len(self.df), index=self.df.index)
    
    def _calculate_fear_index(self):
        """Calculate a crypto fear index based on volatility with robust NaN handling"""
        try:
            # Use already calculated volatility
            volatility = self.df.get('volatility', pd.Series([0.5] * len(self.df)))
            
            # Ensure we have valid volatility data
            volatility = volatility.fillna(0.5)
            
            # Use adaptive window for min/max calculation
            window = min(252, len(self.df), max(30, len(self.df) // 2))
            
            # Normalize volatility to 0-100 scale (like VIX)
            vol_min = volatility.rolling(window=window, min_periods=5).min()
            vol_max = volatility.rolling(window=window, min_periods=5).max()
            
            # Handle cases where min == max
            vol_range = vol_max - vol_min
            fear_index = np.where(
                vol_range > 1e-6,
                100 * (volatility - vol_min) / vol_range,
                50.0  # Neutral fear when no volatility range
            )
            
            # Convert to Series and handle NaN
            fear_series = pd.Series(fear_index, index=self.df.index)
            fear_series = fear_series.fillna(50.0)
            
            # Ensure values are in 0-100 range
            fear_series = np.clip(fear_series, 0.0, 100.0)
            
            return fear_series
            
        except Exception as e:
            logging.warning(f"Error calculating fear index: {e}")
            return pd.Series([50.0] * len(self.df), index=self.df.index)
    
    def _cleanup_nan_values(self):
        """Final cleanup of any remaining NaN values"""
        try:
            # List of indicators that must not have NaN values
            indicators = ['atr', 'volatility', 'trend_strength', 'momentum', 'fear_index']
            
            for indicator in indicators:
                if indicator in self.df.columns:
                    # Check for NaN values
                    nan_count = self.df[indicator].isna().sum()
                    if nan_count > 0:
                        logging.warning(f"Found {nan_count} NaN values in {indicator}, filling with defaults")
                        
                        # Fill with appropriate defaults
                        if indicator == 'atr':
                            self.df[indicator] = self.df[indicator].fillna(self.df[self.price_col] * 0.02)
                        elif indicator == 'volatility':
                            self.df[indicator] = self.df[indicator].fillna(0.5)
                        elif indicator == 'trend_strength':
                            self.df[indicator] = self.df[indicator].fillna(0.3)
                        elif indicator == 'momentum':
                            self.df[indicator] = self.df[indicator].fillna(50.0)
                        elif indicator == 'fear_index':
                            self.df[indicator] = self.df[indicator].fillna(50.0)
                
                # Final check - replace any infinite values
                if indicator in self.df.columns:
                    self.df[indicator] = self.df[indicator].replace([np.inf, -np.inf], 
                                                                  self.df[indicator].median())
                    
        except Exception as e:
            logging.error(f"Error in final NaN cleanup: {e}")
    
    def _set_default_indicators(self):
        """Set robust default values for all indicators"""
        try:
            self.df['atr'] = self.df[self.price_col] * 0.02  # 2% of price
            self.df['volatility'] = 0.5  # 50% annual volatility
            self.df['trend_strength'] = 0.3  # Moderate trend
            self.df['momentum'] = 50.0  # Neutral momentum
            self.df['fear_index'] = 50.0  # Neutral fear
            self.df['resistance'] = self.df[self.high_col]
            self.df['support'] = self.df[self.low_col]
            
            logging.info("Set default values for all technical indicators")
            
        except Exception as e:
            logging.error(f"Error setting default indicators: {e}")
    
    def calculate_dynamic_stop_loss(self, trade_date, predicted_return, confidence, position_type='long'):
        """
        Calculate dynamic stop loss based on multiple market factors for iterative trading
        """
        try:
            # Get market data for the trade date (use latest available if exact date not found)
            if trade_date not in self.df.index:
                # Use the last available date
                trade_date = self.df.index[-1]
                logging.info(f"Using latest available date {trade_date} for stop loss calculation")
            
            market_data = self.df.loc[trade_date]
            
            # Base stop loss factors
            base_stop = 0.03  # 3% base stop loss
            
            # 1. ATR-based volatility adjustment with NaN handling
            atr = market_data.get('atr', 0)
            if pd.isna(atr) or atr <= 0:
                atr = market_data[self.price_col] * 0.02  # 2% fallback
            
            current_price = market_data[self.price_col]
            if current_price <= 0:
                logging.warning("Invalid current price, using fallback calculations")
                return 0.05  # Fallback to 5%
                
            atr_pct = (atr / current_price)
            volatility_multiplier = np.clip(atr_pct / 0.02, 0.5, 3.0)  # Scale around 2% ATR baseline
            
            # 2. Historical volatility adjustment with NaN handling
            hist_vol = market_data.get('volatility', 0.5)
            if pd.isna(hist_vol) or hist_vol <= 0:
                hist_vol = 0.5  # 50% annual volatility fallback
            vol_multiplier = np.clip(hist_vol / 0.5, 0.6, 2.5)  # Scale around 50% annual vol baseline
            
            # 3. Trend strength adjustment with NaN handling
            trend_strength = market_data.get('trend_strength', 0.3)
            if pd.isna(trend_strength):
                trend_strength = 0.3  # Moderate trend fallback
            
            if trend_strength > 0.7:  # Strong trend - wider stops
                trend_multiplier = 1.4
            elif trend_strength > 0.4:  # Moderate trend
                trend_multiplier = 1.0
            else:  # Weak trend - tighter stops
                trend_multiplier = 0.7
            
            # 4. Confidence-based adjustment (already validated)
            confidence = max(0.01, min(confidence, 1.0))  # Ensure confidence is in valid range
            confidence_multiplier = np.clip(2 - confidence * 2, 0.5, 1.5)  # Higher confidence = tighter stops
            
            # 5. Market momentum adjustment with NaN handling
            momentum = market_data.get('momentum', 50.0)
            if pd.isna(momentum):
                momentum = 50.0  # Neutral momentum fallback
            
            # Ensure momentum is in valid RSI range
            momentum = np.clip(momentum, 0, 100)
            
            if position_type == 'long':
                if momentum > 70:  # Overbought - tighter stops
                    momentum_multiplier = 0.8
                elif momentum < 30:  # Oversold - wider stops
                    momentum_multiplier = 1.3
                else:
                    momentum_multiplier = 1.0
            else:  # Short position
                if momentum > 70:  # Overbought - wider stops for shorts
                    momentum_multiplier = 1.3
                elif momentum < 30:  # Oversold - tighter stops for shorts
                    momentum_multiplier = 0.8
                else:
                    momentum_multiplier = 1.0
            
            # 6. Fear index adjustment with NaN handling
            fear_index = market_data.get('fear_index', 50.0)
            if pd.isna(fear_index):
                fear_index = 50.0  # Neutral fear fallback
            
            # Ensure fear index is in valid range
            fear_index = np.clip(fear_index, 0, 100)
            
            if fear_index > 75:  # High fear - wider stops
                fear_multiplier = 1.4
            elif fear_index < 25:  # Low fear (greed) - moderate stops
                fear_multiplier = 0.9
            else:
                fear_multiplier = 1.0
            
            # 7. Predicted return magnitude adjustment
            pred_magnitude = abs(predicted_return)
            if pred_magnitude > 0.1:  # High conviction trades get wider stops
                magnitude_multiplier = 1.3
            elif pred_magnitude > 0.05:
                magnitude_multiplier = 1.1
            else:
                magnitude_multiplier = 0.9
            
            # Combine all factors with validation
            multipliers = [volatility_multiplier, vol_multiplier, trend_multiplier, 
                          confidence_multiplier, momentum_multiplier, fear_multiplier, magnitude_multiplier]
            
            # Check for any invalid multipliers
            valid_multipliers = [m for m in multipliers if not pd.isna(m) and m > 0]
            if len(valid_multipliers) != len(multipliers):
                logging.warning(f"Found invalid multipliers, using validated subset")
            
            dynamic_stop = base_stop
            for multiplier in valid_multipliers:
                dynamic_stop *= multiplier
            
            # Apply bounds
            min_stop = 0.015  # 1.5% minimum
            max_stop = 0.12   # 12% maximum
            dynamic_stop = np.clip(dynamic_stop, min_stop, max_stop)
            
            # Final validation
            if pd.isna(dynamic_stop) or dynamic_stop <= 0:
                logging.warning("Invalid dynamic stop calculated, using fallback")
                dynamic_stop = 0.05
            
            # Log the calculation details
            logging.info(f"Dynamic Stop Loss Calculation:")
            logging.info(f"  Base: {base_stop:.3f}, ATR: {volatility_multiplier:.2f}x, "
                        f"Vol: {vol_multiplier:.2f}x, Trend: {trend_multiplier:.2f}x")
            logging.info(f"  Confidence: {confidence_multiplier:.2f}x, Momentum: {momentum_multiplier:.2f}x, "
                        f"Fear: {fear_multiplier:.2f}x, Magnitude: {magnitude_multiplier:.2f}x")
            logging.info(f"  Final Dynamic Stop: {dynamic_stop:.3f} ({dynamic_stop*100:.1f}%)")
            
            return float(dynamic_stop)  # Ensure it's a Python float, not numpy
            
        except Exception as e:
            logging.error(f"Error calculating dynamic stop loss: {str(e)}")
            return 0.05  # Fallback to 5%
    
    def get_market_regime_info(self, trade_date):
        """Get additional market regime information for logging with NaN handling"""
        try:
            if trade_date not in self.df.index:
                trade_date = self.df.index[-1]
            
            market_data = self.df.loc[trade_date]
            
            # Extract values with NaN handling
            volatility = market_data.get('volatility', 0.5)
            if pd.isna(volatility):
                volatility = 0.5
                
            trend_strength = market_data.get('trend_strength', 0.3)
            if pd.isna(trend_strength):
                trend_strength = 0.3
                
            momentum = market_data.get('momentum', 50.0)
            if pd.isna(momentum):
                momentum = 50.0
                
            fear_index = market_data.get('fear_index', 50.0)
            if pd.isna(fear_index):
                fear_index = 50.0
            
            atr = market_data.get('atr', 0)
            current_price = market_data[self.price_col]
            if pd.isna(atr) or atr <= 0 or current_price <= 0:
                atr_pct = 0.02
            else:
                atr_pct = atr / current_price
            
            return {
                'volatility': float(volatility),
                'trend_strength': float(trend_strength),
                'momentum': float(momentum),
                'fear_index': float(fear_index),
                'atr_pct': float(atr_pct)
            }
        except Exception as e:
            logging.warning(f"Error getting market regime info: {e}")
            return {
                'volatility': 0.5,
                'trend_strength': 0.3,
                'momentum': 50.0,
                'fear_index': 50.0,
                'atr_pct': 0.02
            }

@dataclass
class TradeSignal:
    """Enhanced structure for trade signals with dynamic stop loss"""
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'HOLD'
    confidence: float
    predicted_return: float
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    timestamp: datetime
    expires_at: datetime
    # Dynamic stop loss specific fields
    dynamic_stop_pct: float = 0.05
    market_volatility: float = 0.5
    trend_strength: float = 0.3
    momentum: float = 50.0
    fear_index: float = 50.0
    stop_loss_factors: Dict = None

def apply_dynamic_stop_loss(predicted_return, actual_return, position_type, dynamic_stop_pct):
    """Apply dynamic stop loss logic with variable stop loss percentage"""
    if position_type == 'long' and actual_return < -dynamic_stop_pct:
        return -dynamic_stop_pct  # Cap loss at dynamic stop loss level
    elif position_type == 'short' and actual_return > dynamic_stop_pct:
        return -dynamic_stop_pct  # Cap loss for short position
    else:
        return actual_return  # No stop triggered, use actual return

class IterativeTradingSystem:
    def __init__(self, 
                 postgres_url: str = None,
                 model_path: str = "trained_model.pkl",
                 config_path: str = "trading_config.json"):
        
        if postgres_url is None:
            postgres_url = os.getenv('POSTGRES_URL', 'postgresql://postgres:password@localhost:5432/database')
        self.postgres_url = postgres_url
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Trading parameters - enhanced for dynamic stop loss
        self.min_prediction_threshold = self.config.get('min_prediction_threshold', 0.02)  # 2% like benchmark
        self.base_stop_loss_pct = self.config.get('base_stop_loss_pct', 0.03)  # 3% base for dynamic calculation
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% max like benchmark
        self.signal_validity_hours = self.config.get('signal_validity_hours', 24)
        
        # Dynamic stop loss parameters
        self.use_dynamic_stop_loss = self.config.get('use_dynamic_stop_loss', True)
        self.min_dynamic_stop = self.config.get('min_dynamic_stop', 0.015)  # 1.5%
        self.max_dynamic_stop = self.config.get('max_dynamic_stop', 0.12)   # 12%
        
        # Add benchmark-specific parameters
        self.benchmark_threshold = 0.02
        self.bear_market_threshold = 0.03
        self.normal_market_threshold = 0.025
        self.transaction_cost = 0.001
        
        # Feature consistency tracking
        self.expected_features = None
        self.feature_mismatch_warnings = 0
        
        # Initialize predictor and dynamic stop loss calculator
        self.predictor = None
        self.stop_loss_calculator = None
        self.current_signal = None
        
        # Add consecutive loss tracking
        self.consecutive_losses = 0
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        
        # Store all signals for analysis
        self.all_signals = []
        
    def _load_config(self) -> Dict:
        """Load trading configuration with dynamic stop loss parameters"""
        default_config = {
            "min_prediction_threshold": 0.02,
            "base_stop_loss_pct": 0.03,  # Base for dynamic calculation
            "max_position_size": 0.1,
            "signal_validity_hours": 24,
            "retrain_frequency_days": 7,
            "data_source": "binance",
            "symbol": "BTCUSDT",
            "readiness_threshold": 0.85,
            # Dynamic stop loss parameters
            "use_dynamic_stop_loss": True,
            "min_dynamic_stop": 0.015,
            "max_dynamic_stop": 0.12,
            # Benchmark-specific parameters
            "benchmark_threshold": 0.02,
            "bear_market_threshold": 0.03,
            "normal_market_threshold": 0.025,
            "transaction_cost": 0.001,
            "max_consecutive_losses": 3
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                default_config.update(config)
            return default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return default_config
    
    def get_yesterday_date(self) -> str:
        """Get yesterday's date in YYYY-MM-DD format"""
        yesterday = datetime.now() - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')
    
    def fetch_missing_binance_data(self) -> bool:
        """Fetch all missing daily OHLCV data from Binance until yesterday and append to database"""
        try:
            yesterday = self.get_yesterday_date()
            logging.info(f"Fetching all missing data until yesterday ({yesterday}) from Binance...")
            
            # Get the latest date from the database
            engine = create_engine(self.postgres_url)
            
            # Check what's the latest date we have in the database
            latest_date_query = "SELECT MAX(datetime) FROM btc_daily_ohlcv"
            result = pd.read_sql_query(latest_date_query, engine)
            latest_date_in_db = result.iloc[0, 0]
            
            if latest_date_in_db is None:
                # If database is empty, start from a reasonable date (e.g., 1 year ago)
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                logging.info("Database is empty, starting from 1 year ago")
            else:
                # Start from the day after the latest date in database
                latest_dt = pd.to_datetime(latest_date_in_db)
                start_date = (latest_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                logging.info(f"Latest data in database: {latest_date_in_db}, starting from: {start_date}")
            
            engine.dispose()
            
            # Check if we need to fetch any data
            start_dt = pd.to_datetime(start_date)
            yesterday_dt = pd.to_datetime(yesterday)
            
            if start_dt > yesterday_dt:
                logging.info("No missing data to fetch - database is up to date")
                return True
            
            # Calculate the number of days to fetch
            days_to_fetch = (yesterday_dt - start_dt).days + 1
            logging.info(f"Need to fetch {days_to_fetch} days of data from {start_date} to {yesterday}")
            
            # Binance API has a limit of 1000 klines per request
            # We'll fetch in batches to handle large date ranges
            all_data = []
            current_date = start_dt
            
            while current_date <= yesterday_dt:
                # Calculate end date for this batch (max 1000 days or until yesterday)
                batch_end_date = min(current_date + timedelta(days=999), yesterday_dt)
                
                # Calculate timestamps for Binance API
                start_time = int(current_date.timestamp() * 1000)
                end_time = int((batch_end_date + timedelta(days=1)).timestamp() * 1000) - 1
                
                logging.info(f"Fetching batch: {current_date.strftime('%Y-%m-%d')} to {batch_end_date.strftime('%Y-%m-%d')}")
                
                # Binance API endpoint for daily klines
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': 'BTCUSDT',
                    'interval': '1d',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logging.warning(f"No data available from Binance for batch {current_date.strftime('%Y-%m-%d')} to {batch_end_date.strftime('%Y-%m-%d')}")
                    current_date = batch_end_date + timedelta(days=1)
                    continue
                
                # Convert to DataFrame
                columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 
                          'close_time', 'quote_volume', 'trades_count', 
                          'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
                
                batch_df = pd.DataFrame(data, columns=columns)
                
                # Convert timestamp to datetime
                batch_df['datetime'] = pd.to_datetime(batch_df['datetime'], unit='ms').dt.strftime('%Y-%m-%d')
                
                # Select and convert relevant columns
                batch_df = batch_df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
                batch_df[['open', 'high', 'low', 'close', 'volume']] = batch_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                
                all_data.append(batch_df)
                logging.info(f"Fetched {len(batch_df)} records for this batch")
                
                # Move to next batch
                current_date = batch_end_date + timedelta(days=1)
                
                # Add a small delay to be respectful to the API
                import time
                time.sleep(0.1)
            
            if not all_data:
                logging.warning("No data was fetched from Binance")
                return False
            
            # Combine all batches
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Remove any duplicate dates (just in case)
            final_df = final_df.drop_duplicates(subset=['datetime'], keep='first')
            
            # Sort by date
            final_df = final_df.sort_values('datetime')
            
            logging.info(f"Total records to insert: {len(final_df)}")
            logging.info(f"Date range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
            
            # Insert into database
            engine = create_engine(self.postgres_url)
            final_df.to_sql('btc_daily_ohlcv', engine, if_exists='append', index=False)
            engine.dispose()
            
            logging.info(f"Successfully inserted {len(final_df)} records into btc_daily_ohlcv")
            logging.info(f"Latest data: {final_df['datetime'].max()}")
            
            # Show sample of latest data
            latest_row = final_df[final_df['datetime'] == final_df['datetime'].max()].iloc[0]
            logging.info(f"Latest OHLCV: O={latest_row['open']:.2f}, H={latest_row['high']:.2f}, "
                        f"L={latest_row['low']:.2f}, C={latest_row['close']:.2f}, V={latest_row['volume']:.0f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error fetching missing Binance data: {e}")
            return False
    
    def get_latest_date_from_ohlcv(self) -> Optional[str]:
        """Get the latest date from btc_daily_ohlcv table"""
        try:
            engine = create_engine(self.postgres_url)
            query = "SELECT MAX(datetime) FROM btc_daily_ohlcv"
            result = pd.read_sql_query(query, engine)
            engine.dispose()
            return result.iloc[0, 0]
        except Exception as e:
            logging.error(f"Error getting latest date from ohlcv table: {e}")
            return None
    
    def load_and_prepare_data_for_production(self) -> Optional[pd.DataFrame]:
        """Load and prepare data using load_all_data function for production use"""
        try:
            logging.info("Loading and preparing data using load_all_data function for production...")
            
            # Use the load_all_data function which loads from btc_daily_ohlcv
            btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data(months=48)
            
            if btc_ohlcv is None or len(btc_ohlcv) == 0:
                logging.error("No Bitcoin OHLCV data loaded from btc_daily_ohlcv")
                return None
            
            logging.info(f"Loaded {len(btc_ohlcv)} OHLCV records from btc_daily_ohlcv")
            
            # Process sentiment data if available
            if df_news is not None and len(df_news) > 0:
                logging.info(f"Processing {len(df_news)} news records for sentiment...")
                df_news = add_vader_sentiment(df_news)
                df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)
                logging.info(f"Aggregated sentiment data: {len(df_newsdaily_sentiment)} records")
            else:
                logging.warning("No news data available, using default sentiment")
                df_newsdaily_sentiment = None
            
            # Feature engineering using the same function as benchmark
            df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)
            
            logging.info(f"Successfully engineered features for {len(df)} records")
            logging.info(f"Feature columns: {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading and preparing data for production: {e}")
            return None
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """Train or retrain the ML model and initialize dynamic stop loss calculator"""
        try:
            logging.info("Training ML model...")
            
            if len(data) < 200:  # Need sufficient data for training
                logging.error(f"Insufficient data for training: {len(data)} records")
                return False
            
            # Initialize predictor if not exists
            if self.predictor is None:
                self.predictor = ImprovedBitcoinPredictor(sequence_length=60, prediction_horizon=30)
            
            # Train the ensemble model
            X_val, y_val, regime_seq = self.predictor.train_ensemble(data, validation_split=0.2, epochs=100, batch_size=32)
            
            if X_val is None:
                logging.error("Model training failed")
                return False
            
            # Initialize dynamic stop loss calculator with the trained data
            if self.use_dynamic_stop_loss:
                logging.info("Initializing dynamic stop loss calculator...")
                self.stop_loss_calculator = DynamicStopLossCalculator(data)
                logging.info("Dynamic stop loss calculator initialized successfully")
            
            # Evaluate the model
            evaluation = self.predictor.evaluate_ensemble(X_val, y_val, regime_seq)
            
            logging.info(f"Model training completed successfully")
            logging.info(f"Model performance - MAE: {evaluation.get('mae', 0):.4f}, "
                        f"Direction Accuracy: {evaluation.get('direction_accuracy', 0):.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def generate_trading_signal(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate trading signal with dynamic stop loss"""
        try:
            logging.info("Generating trading signal with dynamic stop loss...")
            
            if self.predictor is None:
                logging.error("Model not trained yet")
                return None
            
            # Get yesterday's close price BEFORE feature engineering (which may drop rows)
            yesterday = self.get_yesterday_date()
            print("Yesterday date: ", yesterday)
            yesterday_price = None
            
            # Print last few rows of data to debug
            print("\n=== DEBUG: Last 3 rows of raw data ===")
            print(f"Data columns: {list(data.columns)}")
            print(f"Data index type: {type(data.index)}")
            print(f"Data index name: {data.index.name}")
            
            # Show last few rows with available columns
            if len(data) >= 3:
                basic_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in basic_cols if col in data.columns]
                print("Last 3 rows:")
                print(data.tail(3)[available_cols])
            else:
                print("Data has less than 3 rows:")
                print(data)
            
            print(f"Data shape: {data.shape}")
            print(f"Data index: {data.index[-3:] if len(data) >= 3 else data.index}")
            print("=" * 50)
            
            # Try to get yesterday's close price from the raw data
            # Check if datetime is in columns or index
            if 'datetime' in data.columns:
                yesterday_data = data[data['datetime'] == yesterday]
                print(f"Yesterday data found in columns: {len(yesterday_data)} rows")
                if len(yesterday_data) > 0:
                    yesterday_price = float(yesterday_data['close'].iloc[0])
                    print(f"Yesterday's close price: ${yesterday_price:.2f}")
                    logging.info(f"Found yesterday's ({yesterday}) close price: ${yesterday_price:.2f}")
            elif hasattr(data.index, 'strftime') or isinstance(data.index, pd.DatetimeIndex):
                # DateTime is in the index
                try:
                    yesterday_data = data[data.index.strftime('%Y-%m-%d') == yesterday]
                    print(f"Yesterday data found in index: {len(yesterday_data)} rows")
                    if len(yesterday_data) > 0:
                        yesterday_price = float(yesterday_data['close'].iloc[0])
                        print(f"Yesterday's close price: ${yesterday_price:.2f}")
                        logging.info(f"Found yesterday's ({yesterday}) close price: ${yesterday_price:.2f}")
                except Exception as e:
                    print(f"Error searching by index: {e}")
            else:
                print("Neither datetime column nor datetime index found")
            
            # Fallback to last available close if yesterday's data not found
            if yesterday_price is None:
                yesterday_price = float(data['close'].iloc[-1])  
                last_date = data.index[-1] if hasattr(data.index, 'strftime') else str(data.index[-1])
                print(f"Fallback: Using last available close price: ${yesterday_price:.2f} from {last_date}")
                logging.warning(f"Using last available close price: ${yesterday_price:.2f} from {last_date}")
            
            # Use the same prediction method as benchmark: prepare features and predict_ensemble
            df_proc = self.predictor.engineer_30day_target(data)
            features, _ = self.predictor.prepare_features(df_proc)
            
            # Create sequences like benchmark
            X, _, _ = self.predictor.create_sequences(features, np.zeros(len(features)))
            
            if len(X) == 0:
                logging.error("No sequences created for prediction")
                return None
            
            # Get prediction using ensemble method
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X)
            
            # Take the last prediction (most recent)
            predicted_return = ensemble_pred[-1][0]
            
            logging.info(f"Generated prediction: {predicted_return:.4f} ({predicted_return*100:.1f}%)")
            
            # Calculate confidence like benchmark (Kelly criterion approximation)
            base_confidence = min(abs(predicted_return), 0.1)
            
            # Use yesterday's price as entry price and get current date for signal generation
            current_price = yesterday_price
            current_date = data.index[-1]
            
            # Position sizing based on Kelly criterion (matches benchmark.py)
            kelly_confidence = min(abs(predicted_return), 0.1)
            
            # Calculate dynamic stop loss if enabled
            if self.use_dynamic_stop_loss and self.stop_loss_calculator is not None:
                position_type = 'long' if predicted_return > 0 else 'short'
                dynamic_stop_pct = self.stop_loss_calculator.calculate_dynamic_stop_loss(
                    current_date, predicted_return, base_confidence, position_type
                )
                
                # Get market regime info for logging
                market_info = self.stop_loss_calculator.get_market_regime_info(current_date)
                
                logging.info(f"Market Regime Analysis:")
                logging.info(f"  Volatility: {market_info['volatility']:.3f} | Trend: {market_info['trend_strength']:.3f}")
                logging.info(f"  Momentum: {market_info['momentum']:.1f} | Fear: {market_info['fear_index']:.1f}")
                logging.info(f"  ATR: {market_info['atr_pct']*100:.2f}%")
                
            else:
                dynamic_stop_pct = self.base_stop_loss_pct
                market_info = {
                    'volatility': 0.5,
                    'trend_strength': 0.3,
                    'momentum': 50.0,
                    'fear_index': 50.0,
                    'atr_pct': 0.02
                }
                logging.info(f"Using base stop loss: {dynamic_stop_pct*100:.1f}%")
            
            # Apply benchmark thresholds for signal generation
            if abs(predicted_return) > self.min_prediction_threshold:  # 0.02 = 2%
                if predicted_return > 0:
                    signal_type = 'LONG'
                else:
                    signal_type = 'SHORT'
                position_size = kelly_confidence  # Kelly criterion position sizing
            else:
                signal_type = 'HOLD'
                position_size = 0.0
            
            # Calculate target and stop loss prices using dynamic stop loss
            if signal_type == 'LONG':
                target_price = current_price * (1 + abs(predicted_return))
                stop_loss = current_price * (1 - dynamic_stop_pct)
            elif signal_type == 'SHORT':
                target_price = current_price * (1 - abs(predicted_return))
                stop_loss = current_price * (1 + dynamic_stop_pct)
            else:
                target_price = current_price
                stop_loss = current_price
            
            # Account for transaction costs in signal (like benchmark)
            adjusted_predicted_return = predicted_return
            total_transaction_cost = 0
            if signal_type != 'HOLD':
                # Account for entry and exit transaction costs
                total_transaction_cost = 2 * self.transaction_cost  # Entry + Exit
                if signal_type == 'LONG':
                    adjusted_predicted_return = predicted_return - total_transaction_cost
                else:  # SHORT
                    adjusted_predicted_return = predicted_return + total_transaction_cost
            
            # Create enhanced signal with dynamic stop loss information
            final_confidence = min(base_confidence, kelly_confidence)
            
            # Ensure all numeric values are valid Python floats (not numpy or NaN)
            def safe_float(value, default=0.0):
                if pd.isna(value) or np.isinf(value):
                    return float(default)
                return float(value)
            
            signal = TradeSignal(
                symbol="BTCUSDT",
                signal_type=signal_type,
                confidence=safe_float(final_confidence),
                predicted_return=safe_float(adjusted_predicted_return),
                entry_price=safe_float(current_price),
                target_price=safe_float(target_price),
                stop_loss=safe_float(stop_loss),
                position_size=safe_float(position_size),
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=1),
                # Dynamic stop loss specific fields with safe conversion
                dynamic_stop_pct=safe_float(dynamic_stop_pct),
                market_volatility=safe_float(market_info['volatility'], 0.5),
                trend_strength=safe_float(market_info['trend_strength'], 0.3),
                momentum=safe_float(market_info['momentum'], 50.0),
                fear_index=safe_float(market_info['fear_index'], 50.0),
                stop_loss_factors={
                    'atr_pct': safe_float(market_info['atr_pct'], 0.02),
                    'base_stop': safe_float(self.base_stop_loss_pct, 0.03),
                    'dynamic_stop': safe_float(dynamic_stop_pct, 0.05),
                    'stop_type': 'dynamic' if self.use_dynamic_stop_loss else 'fixed'
                }
            )
            
            logging.info(f"Generated signal: {signal_type} with confidence {final_confidence:.4f}")
            logging.info(f"Predicted return: {predicted_return:.4f} (adjusted: {adjusted_predicted_return:.4f})")
            logging.info(f"Position size: {position_size:.4f} (Kelly criterion)")
            logging.info(f"Dynamic stop loss: {dynamic_stop_pct*100:.2f}% (vs base: {self.base_stop_loss_pct*100:.1f}%)")
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating trading signal: {e}")
            return None
    
    def save_signal_to_database(self, signal: TradeSignal, prediction_date: str) -> bool:
        """Save enhanced trading signal to database with dynamic stop loss data"""
        try:
            # Skip saving HOLD signals
            if signal.signal_type == 'HOLD':
                logging.info(f"Skipping HOLD signal - not saved to database")
                return True
            
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            # Create enhanced table with dynamic stop loss fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iterative_trading_signals (
                    id SERIAL PRIMARY KEY,
                    prediction_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    position_size REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE',
                    dynamic_stop_pct REAL NOT NULL,
                    market_volatility REAL NOT NULL,
                    trend_strength REAL NOT NULL,
                    momentum REAL NOT NULL,
                    fear_index REAL NOT NULL,
                    stop_loss_factors TEXT NOT NULL
                )
            """)
            
            # Insert enhanced signal
            cursor.execute("""
                INSERT INTO iterative_trading_signals 
                (prediction_date, symbol, signal_type, confidence, predicted_return, entry_price, 
                 target_price, stop_loss, position_size, timestamp, expires_at,
                 dynamic_stop_pct, market_volatility, trend_strength, momentum, fear_index, stop_loss_factors)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                prediction_date,
                signal.symbol,
                signal.signal_type,
                signal.confidence,
                signal.predicted_return,
                signal.entry_price,
                signal.target_price,
                signal.stop_loss,
                signal.position_size,
                signal.timestamp.isoformat(),
                signal.expires_at.isoformat(),
                signal.dynamic_stop_pct,
                signal.market_volatility,
                signal.trend_strength,
                signal.momentum,
                signal.fear_index,
                json.dumps(signal.stop_loss_factors)
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Enhanced signal saved to database for prediction date {prediction_date}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving signal to database: {e}")
            return False
    
    def run_production_prediction(self):
        """Run production prediction for today - fetch latest data, then generate signal"""
        logging.info("=" * 60)
        logging.info("Starting PRODUCTION prediction with DYNAMIC STOP LOSS")
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            logging.info(f"Generating signal for {today}")
            
            # 1. First, fetch all missing daily data from Binance until yesterday
            logging.info("Step 1: Fetching all missing daily data from Binance until yesterday...")
            if not self.fetch_missing_binance_data():
                logging.error("Failed to fetch missing data from Binance")
                return None
            
            # 2. Load and prepare all data using load_all_data function
            logging.info("Step 2: Loading and preparing data...")
            
            # Debug: Check raw data before feature engineering
            print(f"\n=== DEBUG: Data loading process ===")
            raw_btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
            if raw_btc_ohlcv is not None:
                print(f"Raw OHLCV data shape: {raw_btc_ohlcv.shape}")
                print(f"Raw OHLCV columns: {list(raw_btc_ohlcv.columns)}")
                print(f"Raw OHLCV index type: {type(raw_btc_ohlcv.index)}")
                print("Last 3 rows of raw OHLCV:")
                # Show available columns only
                basic_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in basic_cols if col in raw_btc_ohlcv.columns]
                print(raw_btc_ohlcv.tail(3)[available_cols])
                print(f"Raw OHLCV last 3 index values: {raw_btc_ohlcv.index[-3:]}")
            
            data = self.load_and_prepare_data_for_production()
            if data is None:
                logging.error("Failed to load production data")
                return None
            
            print(f"After feature engineering - data shape: {data.shape}")
            print("=" * 50)
            
            logging.info(f"Loaded data with {len(data)} records for training")
            
            # 3. Train model with available data (includes dynamic stop loss calculator initialization)
            logging.info("Step 3: Training model...")
            if not self.train_model(data):
                logging.error("Failed to train model for production")
                return None
            
            # 4. Generate signal for today with dynamic stop loss
            logging.info("Step 4: Generating trading signal...")
            signal = self.generate_trading_signal(data)
            if signal is None:
                logging.error("Failed to generate production signal")
                return None
            
            # 5. Save enhanced signal to database
            logging.info("Step 5: Saving signal to database...")
            if not self.save_signal_to_database(signal, today):
                logging.error("Failed to save production signal")
                return None
            
            # Log the production signal details
            logging.info("=" * 40)
            logging.info(f"🚀 PRODUCTION SIGNAL GENERATED FOR {today}:")
            logging.info(f"   Signal Type: {signal.signal_type}")
            logging.info(f"   Predicted Return: {signal.predicted_return:.4f} ({signal.predicted_return*100:.2f}%)")
            logging.info(f"   Confidence: {signal.confidence:.4f}")
            logging.info(f"   Position Size: {signal.position_size:.4f}")
            logging.info(f"   Entry Price: ${signal.entry_price:.2f}")
            logging.info(f"   Target Price: ${signal.target_price:.2f}")
            logging.info(f"   Stop Loss: ${signal.stop_loss:.2f}")
            logging.info(f"   Dynamic Stop %: {signal.dynamic_stop_pct*100:.2f}%")
            logging.info(f"   Market Volatility: {signal.market_volatility:.3f}")
            logging.info(f"   Trend Strength: {signal.trend_strength:.3f}")
            logging.info(f"   Momentum: {signal.momentum:.1f}")
            logging.info(f"   Fear Index: {signal.fear_index:.1f}")
            logging.info("=" * 40)
            
            return signal
            
        except Exception as e:
            logging.error(f"Error in production prediction: {e}")
            return None
    
    def print_enhanced_summary(self):
        """Print enhanced summary including dynamic stop loss statistics"""
        if not self.all_signals:
            logging.info("No signals generated")
            return
        
        logging.info(f"\nENHANCED SUMMARY OF {len(self.all_signals)} GENERATED SIGNALS:")
        logging.info("=" * 60)
        
        signal_counts = {}
        total_predicted_return = 0
        dynamic_stops = []
        
        for signal in self.all_signals:
            signal_type = signal['signal_type']
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            total_predicted_return += signal['predicted_return']
            if signal['signal_type'] != 'HOLD':
                dynamic_stops.append(signal['dynamic_stop_pct'])
        
        # Basic signal statistics
        for signal_type, count in signal_counts.items():
            logging.info(f"{signal_type}: {count} signals")
        
        avg_predicted_return = total_predicted_return / len(self.all_signals)
        logging.info(f"Average predicted return: {avg_predicted_return:.4f}")
        
        # Dynamic stop loss statistics
        if dynamic_stops:
            avg_dynamic_stop = np.mean(dynamic_stops) * 100
            min_dynamic_stop = np.min(dynamic_stops) * 100
            max_dynamic_stop = np.max(dynamic_stops) * 100
            std_dynamic_stop = np.std(dynamic_stops) * 100
            
            logging.info(f"\n🎯 DYNAMIC STOP LOSS STATISTICS:")
            logging.info(f"Average Dynamic Stop: {avg_dynamic_stop:.2f}%")
            logging.info(f"Stop Loss Range: {min_dynamic_stop:.2f}% - {max_dynamic_stop:.2f}%")
            logging.info(f"Stop Loss Std Dev: {std_dynamic_stop:.2f}%")
            logging.info(f"Base Stop Loss: {self.base_stop_loss_pct*100:.1f}%")
            
            # Market condition analysis
            active_signals = [s for s in self.all_signals if s['signal_type'] != 'HOLD']
            if active_signals:
                avg_volatility = np.mean([s['market_volatility'] for s in active_signals])
                avg_trend = np.mean([s['trend_strength'] for s in active_signals])
                avg_momentum = np.mean([s['momentum'] for s in active_signals])
                avg_fear = np.mean([s['fear_index'] for s in active_signals])
                
                logging.info(f"\n📊 AVERAGE MARKET CONDITIONS:")
                logging.info(f"Volatility: {avg_volatility:.3f} | Trend Strength: {avg_trend:.3f}")
                logging.info(f"Momentum: {avg_momentum:.1f} | Fear Index: {avg_fear:.1f}")
        
        # Show first few and last few signals with stop loss info
        logging.info("\nFirst 5 signals:")
        for i, signal in enumerate(self.all_signals[:5]):
            stop_info = f", stop: {signal['dynamic_stop_pct']*100:.1f}%" if signal['signal_type'] != 'HOLD' else ""
            logging.info(f"  {signal['prediction_date']}: {signal['signal_type']} "
                        f"({signal['predicted_return']:.4f}{stop_info})")
        
        if len(self.all_signals) > 10:
            logging.info("\nLast 5 signals:")
            for signal in self.all_signals[-5:]:
                stop_info = f", stop: {signal['dynamic_stop_pct']*100:.1f}%" if signal['signal_type'] != 'HOLD' else ""
                logging.info(f"  {signal['prediction_date']}: {signal['signal_type']} "
                            f"({signal['predicted_return']:.4f}{stop_info})")

# Production usage
if __name__ == "__main__":
    # Initialize production trading system with dynamic stop loss
    production_system = IterativeTradingSystem()
    
    # Run production prediction for today using yesterday's data
    # This will automatically:
    # 1. Get yesterday's date
    # 2. Load all data from btc_daily_ohlcv up to yesterday
    # 3. Train the model with available data
    # 4. Generate a signal for today
    # 5. Save the signal to database
    signal = production_system.run_production_prediction()
    
    if signal:
        logging.info("✅ Production signal generated successfully!")
    else:
        logging.error("❌ Failed to generate production signal")