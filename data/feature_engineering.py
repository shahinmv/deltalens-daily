import numpy as np
import pandas as pd

def engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment):
    df_daily = (
        btc_ohlcv
        #   .join(daily_oi, how='left')
        #   .join(daily_funding_rate, how='left')
        #   .join(df_newsdaily_sentiment, how='left')
    )
    df = df_daily.copy()
    # Fix index type consistency
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if len(df) < 100:
        raise ValueError("Need at least 100 data points for proper LSTM training")
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    df['open_close_ratio'] = df['open'] / df['close']
    df['volume_avg_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['returns_1d'] = df['close'].pct_change()
    df['returns_3d'] = df['close'].pct_change(3)
    df['returns_7d'] = df['close'].pct_change(7)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    for window in [5, 10, 20]:
        df[f'ma_{window}'] = df['close'].rolling(window).mean()
        df[f'price_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_normalized'] = df['macd'] / df['close']
    df['macd_signal_normalized'] = df['macd_signal'] / df['close']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = df['rsi'] / 100
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['volatility_10'] = df['returns_1d'].rolling(10).std()
    df['volatility_20'] = df['returns_1d'].rolling(20).std()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'].pct_change()
    # df['vader_ma_3'] = df['avg_vader_compound'].rolling(3).mean()
    # df['vader_ma_7'] = df['avg_vader_compound'].rolling(7).mean()
    # df['article_count_norm'] = df['article_count'] / df['article_count'].rolling(30).mean()
    # df['funding_rate_ma'] = df['funding_rate'].rolling(7).mean()
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # ADDED: Bearish-specific features to boost SHORT signal generation
    
    # 1. Bearish divergence indicators - RSI vs Price momentum divergence
    rsi_change = df['rsi'].pct_change()
    df['price_rsi_divergence'] = df['returns_1d'] / (rsi_change + 1e-8)
    
    # 2. Volume-weighted selling pressure
    selling_pressure_raw = np.where(df['returns_1d'] < 0, 
                                   df['volume_avg_ratio'] * abs(df['returns_1d']), 0)
    df['selling_pressure'] = selling_pressure_raw
    df['selling_pressure_ma'] = pd.Series(selling_pressure_raw, index=df.index).rolling(5).mean()
    
    # 3. Distribution phase detection (high volume + negative returns)
    distribution_raw = np.where(
        (df['volume_avg_ratio'] > 1.2) & (df['returns_1d'] < -0.01), 1, 0
    )
    df['distribution_signal'] = pd.Series(distribution_raw, index=df.index).rolling(3).sum()
    
    # 4. Bearish MACD signal strength
    df['macd_bearish'] = np.where(
        (df['macd'] < df['macd_signal']) & (df['macd'] < 0), 1, 0
    )
    
    # 5. Lower highs pattern detection
    df['lower_highs'] = (
        (df['high'] < df['high'].shift(1)) & 
        (df['high'].shift(1) < df['high'].shift(2))
    ).astype(int)
    
    # 6. Bearish engulfing pattern (simplified)
    df['bearish_engulfing'] = (
        (df['open'] > df['close'].shift(1)) &  # Today opens above yesterday's close
        (df['close'] < df['open'].shift(1)) &  # Today closes below yesterday's open
        (df['returns_1d'] < -0.015)            # Significant negative return
    ).astype(int)
    
    # 7. Fear and greed proxy (inverse of volume momentum)
    df['fear_proxy'] = np.where(
        (df['returns_1d'] < 0) & (df['volume_avg_ratio'] > 1.0),
        df['volume_avg_ratio'] * abs(df['returns_1d']) * 2,  # Amplify fear signal
        0
    )
    
    # 8. Breakdown momentum (price breaking below MA with volume)
    df['breakdown_momentum'] = np.where(
        (df['close'] < df['ma_20']) & (df['volume_avg_ratio'] > 1.1) & (df['returns_1d'] < -0.01),
        abs(df['returns_1d']) * df['volume_avg_ratio'],
        0
    )
    
    # PRODUCTION MODIFICATION: Handle target variables differently to preserve last row
    df['next_close'] = df['close'].shift(-1)
    df['target_return'] = (df['next_close'] - df['close']) / df['close']
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # PRODUCTION MODIFICATION: Don't drop the last row even if target variables are NaN
    # For production, we need the last row for prediction even without target values
    
    # Identify rows where target variables are NaN (typically the last row)
    target_cols = ['next_close', 'target_return', 'target_direction']
    target_nan_mask = df[target_cols].isna().all(axis=1)
    
    # Drop rows with NaN in non-target columns only
    non_target_cols = [col for col in df.columns if col not in target_cols]
    feature_complete_mask = df[non_target_cols].notna().all(axis=1)
    
    # Keep rows that either have complete features OR are the last row with target NaN
    keep_mask = feature_complete_mask | target_nan_mask
    df_clean = df[keep_mask].copy()
    
    # For rows with NaN target values (last row), fill with placeholder values
    # This allows the model to use the row for prediction without training on invalid targets
    for col in target_cols:
        df_clean[col] = df_clean[col].fillna(0)  # Fill target NaN with 0 (won't be used for training)
    
    if len(df_clean) < 50:
        raise ValueError("Not enough clean data after preprocessing")
    
    print(f"PRODUCTION FEATURE ENGINEERING: Kept {len(df_clean)} rows (vs {len(df)} original)")
    print(f"Last row date: {df_clean.index[-1]}")
    
    return df_clean