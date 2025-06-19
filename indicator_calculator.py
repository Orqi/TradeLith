# indicator_calculator.py
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import warnings

# Suppress pandas_ta specific warnings (if any common ones pop up)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of technical indicators using pandas_ta.
    Each indicator call is wrapped in a try-except block for robustness.
    """
    if df is None or df.empty:
        logging.warning("[Indicator Calc] Input DataFrame is empty or None. Skipping indicator calculation.")
        return df

    df_copy = df.copy()

    required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    for col in required_ohlcv:
        if col not in df_copy.columns:
            logging.warning(f"  [Indicator Calc Warning] Missing critical OHLCV column '{col}'. Some indicators may fail.")
            return df_copy
        
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy[col].fillna(method='ffill', inplace=True)
        df_copy[col].fillna(method='bfill', inplace=True)
        df_copy[col].fillna(0, inplace=True)

    def _safe_ta_call(df_target, indicator_name_str, **kwargs):
        """Helper to safely call pandas_ta indicators."""
        try:
            indicator_func = getattr(df_target.ta, indicator_name_str)
            indicator_func(**kwargs, append=True)
        except AttributeError:
            logging.warning(f"  [Indicator Calc Warning] Indicator '{indicator_name_str}' not found or available. Skipping.")
        except Exception as e:
            logging.warning(f"  [Indicator Calc Warning] Error calculating '{indicator_name_str}': {e}. Skipping.")

    # --- Trend Indicators ---
    _safe_ta_call(df_copy, 'sma', length=10)
    _safe_ta_call(df_copy, 'sma', length=20)
    _safe_ta_call(df_copy, 'sma', length=50)
    _safe_ta_call(df_copy, 'ema', length=10)
    _safe_ta_call(df_copy, 'ema', length=20)
    _safe_ta_call(df_copy, 'ema', length=50)
    _safe_ta_call(df_copy, 'macd') # MACD, MACDh, MACDs
    _safe_ta_call(df_copy, 'adx') # ADX, DMN, DMP
    _safe_ta_call(df_copy, 'ichimoku') # ICHIMOKU_CHIKOU, ICHIMOKU_CLOUD_A, ICHIMOKU_CLOUD_B, ICHIMOKU_Kijun, ICHIMOKU_Tenkan
    _safe_ta_call(df_copy, 'aroon') # AroonUp, AroonDown, AroonOscillator
    _safe_ta_call(df_copy, 'dpo') # Detrended Price Oscillator
    _safe_ta_call(df_copy, 'trix') # Trix

    # --- Momentum Indicators ---
    _safe_ta_call(df_copy, 'rsi', length=14)
    _safe_ta_call(df_copy, 'stoch') # STOCHk, STOCHd
    _safe_ta_call(df_copy, 'cci')
    _safe_ta_call(df_copy, 'mfi') # Requires volume
    _safe_ta_call(df_copy, 'mom') # Momentum
    _safe_ta_call(df_copy, 'roc') # Rate of Change
    _safe_ta_call(df_copy, 'ao') # Awesome Oscillator
    _safe_ta_call(df_copy, 'cmo') # Chande Momentum Oscillator

    # --- Volatility Indicators ---
    _safe_ta_call(df_copy, 'bbands', length=20, std=2) # BBL, BBM, BBU, BBB, BBP
    _safe_ta_call(df_copy, 'atr') # Average True Range
    _safe_ta_call(df_copy, 'donchian') # DCe, DCL, DCU
    _safe_ta_call(df_copy, 'true_range') # True Range

    # --- Volume Indicators ---
    has_meaningful_volume = 'volume' in df_copy.columns and df_copy['volume'].sum() > 0
    if has_meaningful_volume:
        _safe_ta_call(df_copy, 'obv')
        _safe_ta_call(df_copy, 'ad') # Accumulation/Distribution Line
        _safe_ta_call(df_copy, 'chv') # Chaikin Money Flow
        _safe_ta_call(df_copy, 'vwap') # VWAP
    else:
        logging.warning(f"  [Indicator Calc Warning] Skipping volume-based indicators due to lack of meaningful volume data.")

    # Clean up column names
    df_copy.columns = [str(col).replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '') for col in df_copy.columns]
    
    # Fill NaNs created by indicators.
    for col in df_copy.columns:
        if df_copy[col].dtype in ['float64', 'int64']:
            df_copy[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            df_copy[col].fillna(method='ffill', inplace=True)
            df_copy[col].fillna(method='bfill', inplace=True)
            if df_copy[col].isnull().any():
                df_copy[col].fillna(0, inplace=True)

    logging.info(f"[Indicator Calc] Indicators calculated. Final shape: {df_copy.shape}")
    nan_counts = df_copy.isnull().sum()
    if nan_counts.sum() > 0:
        logging.warning(f"[Indicator Calc WARNING] NaNs still present in columns after filling: \n{nan_counts[nan_counts > 0].to_string()}")

    return df_copy
