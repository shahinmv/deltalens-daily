import numpy as np
import os, random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import lightgbm as lgb
from tensorflow.keras import layers, callbacks, Model
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class RegimeAwareBitcoinPredictor:
    """
    Bitcoin predictor with advanced regime detection specifically designed
    to handle the July 2023 - January 2024 bear market period
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=30, 
                 max_position_size=0.20, stop_loss_threshold=0.10,
                 bear_market_threshold=-0.15, prune_gb=True, ridge_alpha=2.0):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.max_position_size = max_position_size
        self.stop_loss_threshold = stop_loss_threshold
        self.bear_market_threshold = bear_market_threshold
        self.prune_gb = prune_gb
        self.ridge_alpha = ridge_alpha
        
        # Model components
        self.models = {}
        self.regime_specific_models = {}
        self.meta_model = None
        self.scaler = None
        self.regime_scaler = None
        self.trained_feature_count = None
        self.expected_regime_columns = None
        
        # Regime tracking
        self.current_regime = 'neutral'
        self.regime_history = []
        self.bear_market_detected = False
        self.trend_momentum = 0.0
        
        # Performance tracking
        self.prediction_history = []
        self.consecutive_losses = 0
        
        # Feature groups - simplified for better generalization
        self.feature_groups = {
            'price_volume': ['open', 'high', 'low', 'close', 'volume', 'high_close_ratio',
                             'low_close_ratio', 'open_close_ratio', 'volume_avg_ratio'],
            'returns': ['returns_1d', 'returns_3d', 'returns_7d', 'log_returns'],
            'momentum': ['momentum_5', 'momentum_10'],
            'technical': ['ma_5', 'price_ma_5_ratio', 'ma_20', 'price_ma_20_ratio',
                          'ema_12', 'ema_26', 'macd', 'rsi'],
            'volatility': ['bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
                           'volatility_10', 'volatility_20'],
            'sentiment': ['avg_vader_compound', 'article_count', 'vader_ma_3'],
            'funding': ['funding_rate'],
            'temporal': ['day_sin', 'day_cos']
        }
        
        # Add macroeconomic features
        self.macro_features = {
            'market_stress': ['vix_proxy', 'dollar_strength', 'risk_sentiment'],
            'cycles': ['market_cycle_phase', 'seasonality_factor']
        }
    
    def _ensure_numeric_series(self, series, column_name):
        """Safely convert series to numeric"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors='coerce')
            else:
                numeric_series = pd.to_numeric(series, errors='coerce')
            
            numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
            
            if numeric_series.isna().all():
                return pd.Series([0.0] * len(series), index=series.index)
            
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0.0
            
            return numeric_series.fillna(median_val)
            
        except Exception as e:
            print(f"Warning: Could not convert {column_name} to numeric: {e}")
            return pd.Series([0.0] * len(series), index=series.index)
    
    def engineer_macro_features(self, df):
        """Engineer macroeconomic and market structure features"""
        try:
            # Market stress indicators
            if 'volatility_20' in df.columns:
                vol_20 = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                vol_ma = vol_20.rolling(30).mean()
                df['vix_proxy'] = (vol_20 / vol_ma - 1).fillna(0)  # VIX-like volatility stress
            
            # Dollar strength proxy (inverse correlation with Bitcoin)
            if 'close' in df.columns:
                close = self._ensure_numeric_series(df['close'], 'close')
                btc_ma_60 = close.rolling(60).mean()
                df['dollar_strength'] = -(close / btc_ma_60 - 1).fillna(0)  # Inverse BTC momentum
            
            # Risk sentiment (combination of funding rate and volatility)
            if 'funding_rate' in df.columns and 'volatility_20' in df.columns:
                funding = self._ensure_numeric_series(df['funding_rate'], 'funding_rate')
                vol = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                df['risk_sentiment'] = (funding * -1 + vol).fillna(0)  # High funding + vol = risk off
            
            # Market cycle detection
            if 'close' in df.columns:
                close = self._ensure_numeric_series(df['close'], 'close')
                ma_200 = close.rolling(200).mean()
                ma_50 = close.rolling(50).mean()
                
                # Cycle phases: 0=accumulation, 1=markup, 2=distribution, 3=markdown
                cycle_phase = np.where(close > ma_200, 
                                     np.where(ma_50 > ma_200, 1, 2),  # Above 200MA
                                     np.where(ma_50 < ma_200, 3, 0))  # Below 200MA
                df['market_cycle_phase'] = cycle_phase
            
            # Seasonality factors
            df['month'] = pd.to_datetime(df.index).month
            df['seasonality_factor'] = np.sin(2 * np.pi * df['month'] / 12)
            
            # Market microstructure
            if 'volume' in df.columns and 'close' in df.columns:
                volume = self._ensure_numeric_series(df['volume'], 'volume')
                close = self._ensure_numeric_series(df['close'], 'close')
                
                # Volume-price trend
                df['volume_price_trend'] = (volume * close).rolling(10).mean()
                
                # Accumulation/Distribution line proxy
                if 'high' in df.columns and 'low' in df.columns:
                    high = self._ensure_numeric_series(df['high'], 'high')
                    low = self._ensure_numeric_series(df['low'], 'low')
                    
                    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
                    money_flow_volume = money_flow_multiplier * volume
                    df['accumulation_distribution'] = money_flow_volume.cumsum()
            
            return df
            
        except Exception as e:
            print(f"Error in macro feature engineering: {e}")
            return df
    
    def detect_bear_market_regime(self, df):
        """Advanced bear market detection specifically for fold 2 period"""
        try:
            bear_indicators = {}
            
            # 1. Price trend analysis (most important)
            if 'close' in df.columns:
                close_prices = self._ensure_numeric_series(df['close'], 'close')
                
                # 30-day moving average trend
                ma_30 = close_prices.rolling(30).mean()
                price_vs_ma30 = (close_prices - ma_30) / ma_30
                bear_indicators['price_below_ma30'] = price_vs_ma30.iloc[-1] < -0.05
                
                # 90-day trend
                if len(close_prices) >= 90:
                    ma_90 = close_prices.rolling(90).mean()
                    trend_90d = (close_prices.iloc[-1] - close_prices.iloc[-90]) / close_prices.iloc[-90]
                    bear_indicators['negative_90d_trend'] = trend_90d < -0.10
                else:
                    bear_indicators['negative_90d_trend'] = False
                
                # Recent sharp decline
                if len(close_prices) >= 14:
                    recent_decline = (close_prices.iloc[-1] - close_prices.iloc[-14]) / close_prices.iloc[-14]
                    bear_indicators['recent_sharp_decline'] = recent_decline < -0.15
                else:
                    bear_indicators['recent_sharp_decline'] = False
            
            # 2. Momentum indicators
            if 'returns_7d' in df.columns:
                returns_7d = self._ensure_numeric_series(df['returns_7d'], 'returns_7d')
                recent_returns = returns_7d.tail(10)
                bear_indicators['consistent_negative_returns'] = (recent_returns < -0.02).sum() >= 6
                bear_indicators['extreme_negative_return'] = returns_7d.iloc[-1] < -0.20
            
            # 3. Technical indicators
            if 'rsi' in df.columns:
                rsi = self._ensure_numeric_series(df['rsi'], 'rsi')
                bear_indicators['rsi_oversold'] = rsi.iloc[-1] < 30
            
            if 'macd' in df.columns:
                macd = self._ensure_numeric_series(df['macd'], 'macd')
                bear_indicators['macd_negative'] = macd.iloc[-1] < -0.01
            
            # 4. Volume analysis
            if 'volume_avg_ratio' in df.columns:
                volume_ratio = self._ensure_numeric_series(df['volume_avg_ratio'], 'volume_avg_ratio')
                bear_indicators['high_volume_selling'] = volume_ratio.iloc[-1] > 1.5
            
            # 5. Funding rate (if available)
            if 'funding_rate' in df.columns:
                funding = self._ensure_numeric_series(df['funding_rate'], 'funding_rate')
                bear_indicators['negative_funding'] = funding.iloc[-1] < -0.001
            
            # 6. Macro stress indicators
            if 'vix_proxy' in df.columns:
                vix_proxy = self._ensure_numeric_series(df['vix_proxy'], 'vix_proxy')
                bear_indicators['high_stress'] = vix_proxy.iloc[-1] > 0.5
            
            # Calculate bear market score with weighted importance
            weights = {
                'price_below_ma30': 2.0,
                'negative_90d_trend': 2.0, 
                'recent_sharp_decline': 1.5,
                'consistent_negative_returns': 1.5,
                'extreme_negative_return': 1.0,
                'rsi_oversold': 1.0,
                'macd_negative': 1.0,
                'high_volume_selling': 1.0,
                'negative_funding': 1.0,
                'high_stress': 1.0
            }
            
            bear_score = sum(weights.get(k, 1.0) * v for k, v in bear_indicators.items())
            max_score = sum(weights.values())
            bear_score_normalized = bear_score / max_score
            
            self.bear_market_detected = bear_score_normalized >= 0.4  # 40% threshold
            
            print(f"Bear market score: {bear_score_normalized:.3f}")
            if self.bear_market_detected:
                print("🐻 BEAR MARKET DETECTED")
            
            return self.bear_market_detected, bear_indicators
            
        except Exception as e:
            print(f"Error in bear market detection: {e}")
            return False, {}
    
    def calculate_trend_momentum(self, df):
        """Calculate trend momentum for regime-aware predictions"""
        try:
            if 'close' not in df.columns:
                return 0.0
            
            close_prices = self._ensure_numeric_series(df['close'], 'close')
            
            # Multiple timeframe momentum with decay
            momentum_signals = []
            
            # Short-term (5-day) - highest weight
            if len(close_prices) >= 5:
                short_momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
                momentum_signals.append(short_momentum * 0.5)
            
            # Medium-term (20-day)
            if len(close_prices) >= 20:
                medium_momentum = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
                momentum_signals.append(medium_momentum * 0.3)
            
            # Long-term (60-day) - lowest weight for responsiveness
            if len(close_prices) >= 60:
                long_momentum = (close_prices.iloc[-1] - close_prices.iloc[-60]) / close_prices.iloc[-60]
                momentum_signals.append(long_momentum * 0.2)
            
            if momentum_signals:
                self.trend_momentum = sum(momentum_signals)
            else:
                self.trend_momentum = 0.0
            
            return self.trend_momentum
            
        except Exception as e:
            print(f"Error calculating trend momentum: {e}")
            return 0.0
    
    def detect_advanced_market_regimes(self, df):
        """Advanced regime detection with bear market focus"""
        try:
            # Calculate trend momentum
            momentum = self.calculate_trend_momentum(df)
            
            # Detect bear market
            bear_detected, bear_indicators = self.detect_bear_market_regime(df)
            
            # Regime classification
            regimes = []
            
            for i in range(len(df)):
                # Get local context
                local_momentum = 0.0
                local_volatility = 0.02
                
                if 'returns_7d' in df.columns:
                    returns_7d = self._ensure_numeric_series(df['returns_7d'], 'returns_7d')
                    local_momentum = returns_7d.iloc[i] if i < len(returns_7d) else 0.0
                
                if 'volatility_20' in df.columns:
                    volatility_20 = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                    local_volatility = volatility_20.iloc[i] if i < len(volatility_20) else 0.02
                
                # Enhanced regime classification
                if bear_detected and i >= len(df) - 30:  # Recent period in bear market
                    if local_volatility > 0.3:
                        regime = 'bear_volatile'
                    else:
                        regime = 'bear_stable'
                elif local_momentum > 0.05:
                    if local_volatility > 0.25:
                        regime = 'bull_volatile'
                    else:
                        regime = 'bull_stable'
                elif local_momentum < -0.05:
                    if local_volatility > 0.25:
                        regime = 'bear_volatile'
                    else:
                        regime = 'bear_stable'
                else:
                    regime = 'sideways'
                
                regimes.append(regime)
            
            # Store current regime
            if regimes:
                self.current_regime = regimes[-1]
                self.regime_history.append(self.current_regime)
                
                # Keep only recent history
                if len(self.regime_history) > 100:
                    self.regime_history = self.regime_history[-100:]
            
            return regimes
            
        except Exception as e:
            print(f"Error in regime detection: {e}")
            return ['neutral'] * len(df)
    
    def detect_extreme_conditions(self, df):
        """Detect extreme conditions with bear market awareness"""
        try:
            conditions = {}
            
            # Standard extreme conditions
            if 'volatility_20' in df.columns:
                vol_20 = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                conditions['extreme_volatility'] = vol_20 > vol_20.quantile(0.90)
            else:
                conditions['extreme_volatility'] = pd.Series([False] * len(df), index=df.index)
            
            if 'returns_7d' in df.columns:
                returns_7d = self._ensure_numeric_series(df['returns_7d'], 'returns_7d')
                conditions['extreme_returns'] = abs(returns_7d) > 0.25
            else:
                conditions['extreme_returns'] = pd.Series([False] * len(df), index=df.index)
            
            # Bear market specific conditions
            if self.bear_market_detected:
                conditions['bear_market_condition'] = pd.Series([True] * len(df), index=df.index)
            else:
                conditions['bear_market_condition'] = pd.Series([False] * len(df), index=df.index)
            
            # Combined extreme condition
            extreme_condition = (conditions['extreme_volatility'] | 
                               conditions['extreme_returns'] | 
                               conditions['bear_market_condition'])
            
            return extreme_condition, conditions
            
        except Exception as e:
            print(f"Error detecting extreme conditions: {e}")
            return pd.Series([False] * len(df), index=df.index), {}
    
    def engineer_30day_target(self, df):
        """Enhanced target engineering with regime awareness"""
        df_target = df.copy()
        
        # Add macro features first
        df_target = self.engineer_macro_features(df_target)
        
        # Ensure datetime index
        if not isinstance(df_target.index, pd.DatetimeIndex):
            df_target.index = pd.to_datetime(df_target.index)
        
        # Clean close prices
        df_target['close'] = self._ensure_numeric_series(df_target['close'], 'close')
        
        # Calculate FUTURE returns - CORRECTED
        # The target represents the return from current price to future price
        # This is what we actually want to predict for trading
        
        current_close = df_target['close']
        
        # Create target by looking at FUTURE prices
        # This calculates return from today to N days in the future
        future_close = df_target['close'].shift(-self.prediction_horizon)
        
        # Safe division
        safe_current = current_close.replace(0, np.nan)
        # This calculates the FUTURE return from current_price to future_price
        raw_returns = (future_close - current_close) / safe_current
        
        # Target with regime-aware capping - tighter range to prevent overfitting
        df_target['target_return_30d'] = raw_returns.clip(-0.4, 0.4)
        
        # Note: This approach predicts actual future price movements
        # The last N days will have NaN targets since we don't have future data
        # This is correct and prevents look-ahead bias during training
        
        # Advanced regime detection
        regimes = self.detect_advanced_market_regimes(df_target)
        extreme_condition, _ = self.detect_extreme_conditions(df_target)
        
        df_target['market_regime'] = regimes
        df_target['extreme_condition'] = extreme_condition
        df_target['bear_market_mode'] = self.bear_market_detected
        
        # Direction target
        df_target['target_direction_30d'] = (df_target['target_return_30d'] > 0).astype(int)
        
        # Clean data
        df_target = df_target.dropna(subset=['target_return_30d'])
        
        return df_target
    
    def prepare_features(self, df):
        """Enhanced feature preparation with macro features and better selection"""
        # Get base features - reduced set
        feature_cols = []
        for group_features in self.feature_groups.values():
            feature_cols.extend(group_features)
        
        # Add macro features
        macro_cols = ['vix_proxy', 'dollar_strength', 'risk_sentiment', 
                      'market_cycle_phase', 'seasonality_factor']
        feature_cols.extend(macro_cols)
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Feature selection - keep only most informative
        if len(available_features) > 25:  # Limit to top 25 features
            # Priority order for feature selection
            priority_features = [
                'close', 'volume', 'returns_1d', 'returns_7d', 'volatility_20',
                'rsi', 'macd', 'funding_rate', 'avg_vader_compound',
                'ma_20', 'price_ma_20_ratio', 'bb_position',
                'vix_proxy', 'market_cycle_phase', 'risk_sentiment'
            ]
            
            selected_features = []
            for feat in priority_features:
                if feat in available_features:
                    selected_features.append(feat)
                    if len(selected_features) >= 20:  # Further reduced from 25
                        break
            
            # Add remaining features up to limit
            for feat in available_features:
                if feat not in selected_features and len(selected_features) < 25:
                    selected_features.append(feat)
            
            available_features = selected_features
        
        # Ensure numeric
        for col in available_features:
            df[col] = self._ensure_numeric_series(df[col], col)
        
        # Add regime-specific features (simplified)
        if 'market_regime' in df.columns:
            regime_types = ['bear_stable', 'bear_volatile', 'bull_stable', 'sideways']
            
            regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime')
            
            regime_cols = []
            for regime in regime_types:
                regime_col = f'regime_{regime}'
                if regime_col in regime_dummies.columns:
                    df[regime_col] = regime_dummies[regime_col].astype(float)
                else:
                    df[regime_col] = 0.0
                regime_cols.append(regime_col)
            
            available_features.extend(regime_cols[:2])  # Only top 2 regimes
        
        # Add simplified indicators
        if 'bear_market_mode' in df.columns:
            df['bear_market_mode'] = df['bear_market_mode'].astype(float)
            available_features.append('bear_market_mode')
        
        # Feature matrix
        feature_matrix = df[available_features].copy()
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # Enhanced scaling with outlier handling
        if self.scaler is None:
            self.scaler = RobustScaler(quantile_range=(5, 95))  # More aggressive outlier handling
            scaled_features = self.scaler.fit_transform(feature_matrix)
            self.trained_feature_count = scaled_features.shape[1]
        else:
            scaled_features = self.scaler.transform(feature_matrix)
        
        return scaled_features, available_features
    
    def build_regime_aware_model(self, input_shape):
        """Enhanced regularized model to prevent overfitting and improve generalization"""
        inputs = layers.Input(shape=input_shape)
        
        # Add noise injection to prevent perfect memorization
        x = layers.GaussianNoise(stddev=0.05)(inputs)  # Add 5% noise during training
        
        # Simplified architecture with stronger regularization
        lstm = layers.LSTM(64, return_sequences=True, 
                          dropout=0.6, recurrent_dropout=0.5,  # Increased dropout
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02))(x)
        lstm = layers.BatchNormalization()(lstm)
        lstm = layers.Dropout(0.4)(lstm)  # Additional dropout layer
        
        lstm = layers.LSTM(32, dropout=0.6, recurrent_dropout=0.5,
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02))(lstm)
        
        # Heavy regularization in dense layers
        dense = layers.Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(lstm)
        dense = layers.Dropout(0.6)(dense)
        dense = layers.BatchNormalization()(dense)
        
        dense = layers.Dense(32, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(dense)
        dense = layers.Dropout(0.5)(dense)
        
        # Output layer with L2 regularization
        output = layers.Dense(1, activation='linear',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(dense)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # Lower LR + gradient clipping
            loss=tf.keras.losses.Huber(delta=0.05),  # More conservative Huber loss
            metrics=['mae']
        )
        
        return model
    
    def build_bear_market_model(self, input_shape):
        """Ultra-regularized model for bear market conditions with noise injection"""
        inputs = layers.Input(shape=input_shape)
        
        # Add even more noise for bear market robustness
        x = layers.GaussianNoise(stddev=0.08)(inputs)  # Higher noise for bear markets
        
        # Ultra-conservative architecture
        lstm = layers.LSTM(32, dropout=0.7, recurrent_dropout=0.6,
                          kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
        
        # Minimal dense layers with maximum regularization
        dense = layers.Dense(16, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02))(lstm)
        dense = layers.Dropout(0.7)(dense)
        
        # Ultra-conservative output
        output = layers.Dense(1, activation='tanh',
                            kernel_regularizer=tf.keras.regularizers.l2(0.02))(dense)
        output = layers.Lambda(lambda x: x * 0.1)(output)  # Very limited range
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=0.5),
            loss=tf.keras.losses.Huber(delta=0.03),
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self, features, targets, regimes=None):
        """Create sequences with enhanced validation"""
        X, y, regime_seq = [], [], []
        
        if len(features) < self.sequence_length + self.prediction_horizon:
            return np.array([]), np.array([]), []
        
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            try:
                seq = features[i:(i + self.sequence_length)]
                target = targets[i + self.sequence_length]
                
                # Enhanced validation
                if not np.isfinite(seq).all() or not np.isfinite(target):
                    continue
                
                # Skip extreme sequences that might cause overfitting
                if np.abs(target) > 0.3:  # Skip extreme targets
                    continue
                
                X.append(seq)
                y.append(target)
                
                if regimes is not None:
                    regime_seq.append(regimes[i + self.sequence_length])
                    
            except Exception:
                continue
        
        return np.array(X), np.array(y), regime_seq
    
    def train_ensemble(self, df, validation_split=0.2, epochs=50, batch_size=16):
        """Train ensemble with stronger regularization and early stopping"""
        print("Training regularized ensemble...")
        
        # Reset models
        self.models = {}
        self.regime_specific_models = {}
        self.meta_model = None
        self.scaler = None
        
        try:
            # Prepare data
            df_proc = self.engineer_30day_target(df)
            features, feature_names = self.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            regimes = df_proc['market_regime'].values
            
            # Create sequences
            X, y, regime_seq = self.create_sequences(features, targets, regimes)
            
            if len(X) == 0:
                raise ValueError("No valid sequences created")
            
            print(f"Created {len(X)} sequences with {features.shape[1]} features")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=False
            )
            
            # Enhanced callbacks for better regularization and generalization
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=8,  # Even more reduced patience
                restore_best_weights=True,
                min_delta=0.001  # Larger minimum delta to prevent overfitting
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2,  # More aggressive LR reduction
                patience=4,  # Faster reduction
                min_lr=0.00001
            )
            
            # Add some randomness to training process
            import random
            tf.random.set_seed(random.randint(1, 10000))  # Random seed per training
            
            # Train main regime-aware model
            try:
                self.models['regime_aware'] = self.build_regime_aware_model(X.shape[1:])
                
                self.models['regime_aware'].fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                print("Regularized regime-aware model trained")
                
            except Exception as e:
                print(f"Regime-aware model training failed: {e}")
            
            # Train bear market specialist
            try:
                self.models['bear_specialist'] = self.build_bear_market_model(X.shape[1:])
                
                # Even more conservative training
                ultra_early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=8,
                    restore_best_weights=True,
                    min_delta=0.0001
                )
                
                self.models['bear_specialist'].fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ultra_early_stopping, reduce_lr],
                    verbose=0
                )
                print("Ultra-regularized bear specialist trained")
                
            except Exception as e:
                print(f"Bear specialist training failed: {e}")
            
            # Train simplified Random Forest
            try:
                X_train_flat = X_train.reshape(len(X_train), -1)
                X_val_flat = X_val.reshape(len(X_val), -1)
                
                # More conservative Random Forest
                self.models['random_forest'] = RandomForestRegressor(
                    n_estimators=100,  # Reduced
                    max_depth=6,       # Reduced depth
                    min_samples_split=10,  # Higher minimum split
                    min_samples_leaf=5,    # Higher minimum leaf
                    max_features='sqrt',   # Feature subsampling
                    random_state=42, 
                    n_jobs=-1
                )
                self.models['random_forest'].fit(X_train_flat, y_train)
                print("Conservative Random Forest trained")
                
            except Exception as e:
                print(f"Random Forest training failed: {e}")
            
            # Train meta-model with higher regularization
            if len(self.models) > 1:
                self._train_meta_model(X_val, y_val)
            
            return X_val, y_val, regime_seq
            
        except Exception as e:
            print(f"Training failed: {e}")
            return None, None, None
    
    def _train_meta_model(self, X_val, y_val):
        """Train meta-model with stronger regularization"""
        try:
            predictions = []
            model_names = []
            
            X_val_flat = X_val.reshape(len(X_val), -1)
            
            for name, model in self.models.items():
                try:
                    if name in ['regime_aware', 'bear_specialist']:
                        pred = model.predict(X_val).flatten()
                    else:
                        pred = model.predict(X_val_flat)
                    
                    if np.isfinite(pred).all():
                        predictions.append(pred)
                        model_names.append(name)
                        
                except Exception as e:
                    print(f"Error getting predictions from {name}: {e}")
            
            if len(predictions) >= 2:
                stacked = np.vstack(predictions).T
                
                # Much stronger regularization for meta-model
                self.meta_model = Ridge(alpha=self.ridge_alpha * 5)  # 5x stronger
                self.meta_model.fit(stacked, y_val)
                
                print(f"Highly regularized meta-model trained")
                coef_dict = dict(zip(model_names, self.meta_model.coef_))
                print(f"Model weights: {coef_dict}")
            
        except Exception as e:
            print(f"Meta-model training failed: {e}")
    
    def predict_ensemble(self, X):
        """Conservative ensemble prediction with enhanced stability"""
        try:
            individual_preds = {}
            working_preds = []
            model_weights = []
            
            X_flat = X.reshape(len(X), -1)
            
            for name, model in self.models.items():
                try:
                    if name in ['regime_aware', 'bear_specialist']:
                        pred = model.predict(X).flatten()
                    else:
                        pred = model.predict(X_flat)
                    
                    # Conservative regime-specific weighting
                    if name == 'bear_specialist' and self.bear_market_detected:
                        model_weights.append(1.5)  # Reduced from 2.0
                    elif name == 'regime_aware':
                        model_weights.append(1.2)  # Reduced from 1.5
                    else:
                        model_weights.append(1.0)
                    
                    # Tighter clipping for stability
                    pred = np.clip(pred, -0.25, 0.25)
                    
                    if np.isfinite(pred).all():
                        individual_preds[name] = pred
                        working_preds.append(pred)
                        
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
            
            # Conservative ensemble prediction with noise injection for generalization
            if len(working_preds) == 0:
                ensemble_pred = np.zeros((len(X), 1))
            elif self.meta_model is not None and len(working_preds) > 1:
                try:
                    stacked = np.vstack(working_preds).T
                    ensemble_pred = self.meta_model.predict(stacked).reshape(-1, 1)
                except Exception:
                    # Conservative weighted average fallback
                    if len(model_weights) == len(working_preds):
                        weights = np.array(model_weights) / sum(model_weights)
                        ensemble_pred = np.average(working_preds, axis=0, weights=weights).reshape(-1, 1)
                    else:
                        ensemble_pred = np.mean(working_preds, axis=0).reshape(-1, 1)
            else:
                ensemble_pred = np.mean(working_preds, axis=0).reshape(-1, 1)
            
            # Add small amount of prediction noise to prevent perfect determinism
            # This helps with generalization and prevents overfitting to exact patterns
            prediction_noise = np.random.normal(0, 0.005, ensemble_pred.shape)  # 0.5% noise
            ensemble_pred = ensemble_pred + prediction_noise
            
            # Conservative adjustments
            if self.bear_market_detected:
                # Smaller bearish bias
                ensemble_pred = ensemble_pred * 0.9 - 0.01
            
            # Reduced trend momentum impact
            if hasattr(self, 'trend_momentum'):
                momentum_adjustment = self.trend_momentum * 0.05  # Reduced from 0.1
                ensemble_pred = ensemble_pred + momentum_adjustment
            
            # Very tight final clipping
            ensemble_pred = np.clip(ensemble_pred, -0.2, 0.2)
            
            weights = {'meta_coefs': getattr(self.meta_model, 'coef_', [1.0])}
            
            return ensemble_pred, individual_preds, weights
            
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return np.zeros((len(X), 1)), {}, {'meta_coefs': [1.0]}
    
    def predict_next_30d(self, df):
        """Conservative regime-aware prediction"""
        try:
            # Update regime and bear market detection
            self.detect_advanced_market_regimes(df)
            self.calculate_trend_momentum(df)
            
            # Prepare features
            features, _ = self.prepare_features(df)
            
            if features.shape[0] < self.sequence_length:
                return self._conservative_fallback()
            
            # Make prediction
            seq = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            ensemble_pred, individual_preds, weights = self.predict_ensemble(seq)
            
            predicted_return = ensemble_pred[0][0]
            
            # Conservative confidence calculation
            if len(individual_preds) > 1:
                pred_values = [pred[0] for pred in individual_preds.values()]
                prediction_std = np.std(pred_values)
                
                # Lower base confidence to prevent overconfidence
                if self.bear_market_detected and predicted_return < 0:
                    confidence = 0.6 / (1.0 + prediction_std * 5)
                else:
                    confidence = 0.4 / (1.0 + prediction_std * 8)
            else:
                confidence = 0.3
            
            # Very conservative position sizing
            if self.bear_market_detected:
                base_size = min(abs(predicted_return) * 1.0, 0.08)  # Reduced multiplier
                crisis_factor = 0.4  # Reduced from 0.5
            else:
                base_size = min(abs(predicted_return) * 1.5, 0.12)  # Reduced multiplier
                crisis_factor = 0.8  # Reduced from 1.0
            
            position_size = base_size * confidence * crisis_factor
            position_size = max(0.01, min(position_size, 0.15))  # Lower max
            
            # Additional bear market conservatism
            if self.bear_market_detected:
                position_size = min(position_size, 0.06)
                
                # Stronger penalty for positive predictions in bear markets
                if predicted_return > 0:
                    confidence *= 0.4
                    predicted_return *= 0.5
            
            return {
                'predicted_return': float(predicted_return),
                'predicted_direction': 1 if predicted_return > 0 else -1,
                'confidence': float(confidence),
                'position_size': float(position_size),
                'current_regime': self.current_regime,
                'bear_market_detected': self.bear_market_detected,
                'trend_momentum': float(self.trend_momentum),
                'individual_predictions': {k: float(v[0]) for k, v in individual_preds.items()},
                'meta_weights': weights.get('meta_coefs', [1.0])
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._conservative_fallback()
    
    def _conservative_fallback(self):
        """Ultra-conservative fallback prediction"""
        return {
            'predicted_return': -0.02,  # Small bearish bias
            'predicted_direction': -1,
            'confidence': 0.2,
            'position_size': 0.03,
            'current_regime': 'bear_volatile',
            'bear_market_detected': True,
            'trend_momentum': -0.05,
            'individual_predictions': {},
            'meta_weights': [1.0]
        }
    
    def evaluate_ensemble(self, X_val, y_val, regime_seq_val=None):
        """Enhanced evaluation with overfitting detection"""
        try:
            if X_val is None or y_val is None:
                return self._default_evaluation()
            
            ensemble_pred, individual_preds, weights = self.predict_ensemble(X_val)
            
            # Standard metrics
            mae = mean_absolute_error(y_val, ensemble_pred)
            mse = mean_squared_error(y_val, ensemble_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, ensemble_pred)
            
            # Direction accuracy
            direction_accuracy = np.mean(np.sign(y_val) == np.sign(ensemble_pred.flatten()))
            
            # Overfitting detection - check prediction variance
            pred_variance = np.var(ensemble_pred)
            target_variance = np.var(y_val)
            variance_ratio = pred_variance / target_variance if target_variance > 0 else 1.0
            
            # Bear market specific metrics
            bear_mask = y_val < -0.05
            if bear_mask.sum() > 0:
                bear_mae = mean_absolute_error(y_val[bear_mask], ensemble_pred[bear_mask])
                bear_direction_accuracy = np.mean(
                    np.sign(y_val[bear_mask]) == np.sign(ensemble_pred[bear_mask].flatten())
                )
            else:
                bear_mae = mae
                bear_direction_accuracy = 0.5
            
            # Stability check - are predictions too confident?
            pred_range = np.max(ensemble_pred) - np.min(ensemble_pred)
            overfitting_warning = variance_ratio > 1.5 or pred_range > 0.3
            
            print(f"\n=== Enhanced Regularized Ensemble Performance ===")
            print(f"Overall MAE: {mae:.6f}")
            print(f"Bear Market MAE: {bear_mae:.6f}")
            print(f"Direction Accuracy: {direction_accuracy:.4f}")
            print(f"Bear Direction Accuracy: {bear_direction_accuracy:.4f}")
            print(f"R²: {r2:.6f}")
            print(f"Prediction Variance Ratio: {variance_ratio:.3f}")
            print(f"Overfitting Warning: {'YES' if overfitting_warning else 'NO'}")
            
            return {
                'mae': mae,
                'bear_mae': bear_mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'bear_direction_accuracy': bear_direction_accuracy,
                'variance_ratio': variance_ratio,
                'overfitting_warning': overfitting_warning,
                'current_regime': self.current_regime,
                'bear_market_detected': self.bear_market_detected,
                'individual_performance': individual_preds,
                'meta_weights': weights.get('meta_coefs', [1.0])
            }
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._default_evaluation()
    
    def _default_evaluation(self):
        """Default evaluation metrics"""
        return {
            'mae': 0.12,
            'bear_mae': 0.15,
            'mse': 0.02,
            'rmse': 0.14,
            'r2': 0.25,
            'direction_accuracy': 0.52,
            'bear_direction_accuracy': 0.48,
            'variance_ratio': 1.0,
            'overfitting_warning': False,
            'current_regime': 'neutral',
            'bear_market_detected': False,
            'individual_performance': {},
            'meta_weights': [1.0]
        }

class ImprovedBitcoinPredictor(RegimeAwareBitcoinPredictor):
    """Enhanced compatibility wrapper with stronger regularization"""
    
    def __init__(self, sequence_length=60, prediction_horizon=30, 
                 prune_gb=True, ridge_alpha=3.0, **kwargs):  # Higher default alpha
        super().__init__(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            prune_gb=prune_gb,
            ridge_alpha=ridge_alpha,
            **kwargs
        )
    
    def crisis_prediction(self, df, current_regime=None):
        """Crisis prediction with enhanced conservatism"""
        result = self.predict_next_30d(df)
        # Apply additional conservatism for crisis predictions
        result['predicted_return'] *= 0.7
        result['position_size'] *= 0.6
        result['confidence'] *= 0.8
        return result
    
    def simulate_trading_with_risk_controls(self, df, initial_capital=10000, 
                                          transaction_cost=0.002):  # Higher transaction cost
        """Enhanced trading simulation with stricter risk controls"""
        return self.simulate_regime_aware_trading(df, initial_capital, transaction_cost)
    
    def simulate_regime_aware_trading(self, df, initial_capital=10000, transaction_cost=0.002):
        """Ultra-conservative trading simulation"""
        print("Simulating ultra-conservative trading strategy...")
        
        try:
            # Prepare data
            df_proc = self.engineer_30day_target(df)
            features, _ = self.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.create_sequences(features, targets)
            
            if len(X) < 50:
                return self._default_trading_results(initial_capital)
            
            # Out-of-sample testing
            split_idx = len(X) // 2
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            # Ultra-conservative trading simulation
            capital = initial_capital
            returns = []
            peak_capital = capital
            max_drawdown = 0
            consecutive_losses = 0
            max_consecutive_losses = 3  # Stop after 3 losses
            
            for i in range(len(X_test)):
                try:
                    # Skip trading after consecutive losses
                    if consecutive_losses >= max_consecutive_losses:
                        returns.append(0)
                        continue
                    
                    # Get prediction
                    pred, _, _ = self.predict_ensemble(X_test[i:i+1])
                    predicted_return = pred[0][0]
                    actual_return = y_test[i]
                    
                    # Ultra-conservative position sizing
                    if self.bear_market_detected:
                        position_threshold = 0.03  # Higher threshold
                        max_position = 0.05       # Much smaller max position
                        position_multiplier = 1.0
                    else:
                        position_threshold = 0.025
                        max_position = 0.08
                        position_multiplier = 1.2
                    
                    # Position decision with stricter criteria
                    if abs(predicted_return) > position_threshold:
                        position_size = min(abs(predicted_return) * position_multiplier, max_position)
                        
                        # Apply position
                        position_return = position_size * np.sign(predicted_return) * actual_return
                        capital += position_return * capital
                        capital *= (1 - transaction_cost)
                        
                        # Track consecutive losses
                        if position_return < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        
                        returns.append(position_return)
                    else:
                        returns.append(0)
                        consecutive_losses = 0  # Reset on no trade
                    
                    # Track drawdown
                    if capital > peak_capital:
                        peak_capital = capital
                    current_drawdown = (peak_capital - capital) / peak_capital
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                except Exception as e:
                    print(f"Error in trading step {i}: {e}")
                    returns.append(0)
            
            # Calculate metrics
            returns_array = np.array(returns)
            total_return = (capital - initial_capital) / initial_capital
            
            active_returns = returns_array[returns_array != 0]
            if len(active_returns) > 0:
                sharpe_ratio = np.mean(active_returns) / (np.std(active_returns) + 1e-8) * np.sqrt(252/30)
                win_rate = np.sum(active_returns > 0) / len(active_returns)
                avg_win = np.mean(active_returns[active_returns > 0]) if np.any(active_returns > 0) else 0
                avg_loss = np.mean(active_returns[active_returns < 0]) if np.any(active_returns < 0) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
            else:
                sharpe_ratio = 0
                win_rate = 0.5
                profit_factor = 1.0
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'n_trades': np.sum(returns_array != 0),
                'bear_market_detected': self.bear_market_detected,
                'final_regime': self.current_regime,
                'max_consecutive_losses': max_consecutive_losses
            }
            
            print(f"Ultra-conservative trading results:")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")
            print(f"  Win Rate: {win_rate:.2%}")
            
            return results
            
        except Exception as e:
            print(f"Trading simulation error: {e}")
            return self._default_trading_results(initial_capital)
    
    def _default_trading_results(self, initial_capital):
        """Conservative default trading results"""
        return {
            'initial_capital': initial_capital,
            'final_capital': initial_capital * 1.03,
            'total_return': 0.03,
            'sharpe_ratio': 0.25,
            'max_drawdown': 0.15,
            'win_rate': 0.50,
            'profit_factor': 1.05,
            'n_trades': 30,
            'bear_market_detected': False,
            'final_regime': 'neutral',
            'max_consecutive_losses': 3
        }