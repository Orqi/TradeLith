import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score
import joblib
import numpy as np
import os
import warnings
import time
from datetime import datetime, timedelta
import sys
import requests # For Alpha Vantage API
import scipy.signal
from sklearn.linear_model import LinearRegression
import json # For parsing Alpha Vantage API responses
import logging # For better logging control

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information.", category=UserWarning)
warnings.filterwarnings("ignore", message="The behavior of DataFrame.idxmin with a float dtype index is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="The behavior of DataFrame.idxmax with a float dtype index is deprecated.*", category=FutureWarning)


# --- Configuration ---
# Configure logging for better output management
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

STOCK_TICKERS = ["GRSE.NS", "ZENTEC.NS", "YESBANK.NS", "KITEX.NS"]

# Training Data Parameters (for AI model)
TRAIN_DATA_PERIOD = "10y"
TRAIN_DATA_INTERVAL = "1d" # Daily data for training

# Live Data Parameters (for monitoring)
LIVE_DATA_PERIOD = "1y" # Changed to 1 year to ensure enough data (approx 250 trading days) for indicators and patterns
LIVE_DATA_INTERVAL = "1d" # Changed to 1d for simplicity with fundamental/news data consistency, can be 5m for faster updates
                            # but fundamental/news APIs are often daily/snapshot.

CHECK_INTERVAL_SECONDS = 60 # How often to poll for new data (5 minutes)

# Minimum bars needed for indicator calculation AND chart patterns.
# Many indicators require a history (e.g., Ichimoku 52 periods, ADX 14, BBands 20).
# Chart patterns often span multiple bars. Set a sufficiently large value.
MIN_REQUIRED_BARS_FOR_ANALYSIS = 200 # Increased for better robustness

# Target Variable Definition
FUTURE_PERIODS = 1 # Predict 1 period ahead (e.g., next day for daily data, next 5-min for 5m data)
UP_THRESHOLD = 0.005 # 0.5% up
DOWN_THRESHOLD = -0.005 # 0.5% down

# Model Training Parameters
MODEL_DIR = "trained_models"
PARAM_GRID = {
    'n_estimators': [50, 100, 200, 300], # Expanded for wider search
    'max_depth': [5, 10, 15, 20, None], # Expanded for wider search
    'min_samples_split': [2, 5, 10],    # Expanded for wider search
    'min_samples_leaf': [1, 2, 4],      # Expanded for wider search
    'class_weight': ['balanced'] # Good for imbalanced target classes
}

# Trading Strategy Parameters
INITIAL_STOP_LOSS_PERCENTAGE = 0.02 # 2%
TRAILING_STOP_LOSS_PERCENTAGE = 0.015 # 1.5%
TAKE_PROFIT_PERCENTAGE = 0.03 # 3%

# Alpha Vantage API Key - IMPORTANT: REPLACE THIS WITH YOUR ACTUAL KEY
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "DCM1DTDLBSGEANZW")

VERBOSE_MODE = True # Set to False for less console output

# --- Global State Management ---
# Using a dictionary to store state for each ticker.
# For a production bot, this would need to be persistent (e.g., saved to a file/DB).
stock_states = {
    ticker: {
        'last_ai_signal': 'Hold', # Last AI prediction for the stock
        'current_price': None,    # Last fetched price
        'in_buy_position': False, # True if currently holding a BUY position
        'entry_price': None,      # Price at which the BUY position was entered
        'stop_loss_price': None,  # Current stop loss price
        'high_since_entry': None, # Highest price reached since entering a BUY position (for trailing SL)
        'last_notification_price': None # Price at last major notification, to reduce spam
    } for ticker in STOCK_TICKERS
}

trade_log = [] # List to store details of executed trades (for current session only)

# --- Features to always include in the model, even if they are constant or often NaN ---
# These are features whose presence (or consistent absence/neutrality) is important,
# and we don't want them filtered out by `nunique() > 1` or `notna().any()` during training.
ALWAYS_INCLUDED_FEATURES = [
    'pe_ratio', 'market_capitalization', 'book_value_per_share', 'dividend_yield', # Fundamental Data
    'avg_news_sentiment', # News Sentiment
    'has_double_top', 'has_double_bottom', 'has_triple_top', 'has_triple_bottom',
    'has_head_and_shoulders', 'has_inverse_head_and_shoulders',
    'has_rising_wedge', 'has_falling_wedge',
    'has_ascending_triangle', 'has_descending_triangle', 'has_symmetrical_triangle',
    'has_bullish_flag', 'has_bearish_flag', 'has_cup_and_handle' # Chart Patterns
]

# --- Helper Functions for Chart Pattern Detection ---

def find_peaks_and_troughs(prices: pd.Series, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies peaks and troughs in a price series using scipy.signal.find_peaks.
    A peak is a point greater than its 'order' neighbors on both sides.
    A trough is a point less than its 'order' neighbors on both sides.
    Returns indices of peaks and troughs.
    """
    prices = pd.to_numeric(prices, errors='coerce').dropna()
    if prices.empty:
        return np.array([]), np.array([])

    # Use a high enough prominence to avoid minor fluctuations
    # Adjusted prominence slightly to potentially detect more patterns
    peaks, _ = scipy.signal.find_peaks(prices, distance=order, prominence=prices.std() * 0.05)
    troughs, _ = scipy.signal.find_peaks(-prices, distance=order, prominence=prices.std() * 0.05)
    
    return peaks, troughs

def fit_trendline(prices: pd.Series, indices: np.ndarray) -> tuple[float, float, float]:
    """
    Fits a linear trendline to a set of prices at given array indices.
    Returns the slope, intercept, and R-squared value.
    """
    if len(indices) < 2:
        return 0, 0, 0 # Cannot fit a line with less than 2 points

    # Filter out indices that are out of bounds for the prices Series
    valid_indices = [idx for idx in indices if 0 <= idx < len(prices)]
    if len(valid_indices) < 2:
        return 0, 0, 0

    x = np.array(valid_indices).reshape(-1, 1)
    y = prices.iloc[valid_indices].values

    if len(np.unique(y)) == 1: # Handle case where all prices are the same
        return 0, y[0], 1.0 # Slope is 0, intercept is the price, R-squared is 1

    model = LinearRegression()
    try:
        model.fit(x, y)
    except ValueError as e:
        # More robust error handling for fitting issues
        if VERBOSE_MODE:
            logging.warning(f"  [Trendline Fit Error] Could not fit trendline: {e}. X shape: {x.shape}, Y shape: {y.shape}. Unique X: {np.unique(x).shape}, Unique Y: {np.unique(y).shape}")
        if len(x) == 1:
            return 0, y[0] if len(y) > 0 else 0, 0
        if len(np.unique(x)) < 2:
             return 0, y[0] if len(y) > 0 else 0, 0
        return 0, 0, 0 # Return zeros to avoid crashing
        
    y_pred = model.predict(x)
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_residual = ((y - y_pred) ** 2).sum()
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    return model.coef_[0], model.intercept_, r_squared

def is_flat(line_slope, tolerance=0.0001):
    """Checks if a trendline is relatively flat."""
    return abs(line_slope) < tolerance

def is_converging(slope1, slope2, is_rising_pattern=True, tolerance=0.001):
    """
    Checks if two trendlines are converging.
    For rising patterns (e.g., rising wedge), upper slope should be steeper than lower slope.
    For falling patterns (e.g., falling wedge), lower slope should be steeper (more negative) than upper.
    """
    if is_rising_pattern: # Both rising, upper steeper (e.g., rising wedge, descending triangle)
        return slope1 > 0 and slope2 > 0 and slope1 > slope2 # upper slope > lower slope (steeper)
    else: # Both falling, lower steeper (e.g., falling wedge, ascending triangle)
        return slope1 < 0 and slope2 < 0 and abs(slope2) > abs(slope1) # abs(lower slope) > abs(upper slope) (steeper)

def is_diverging(slope1, slope2, is_rising_pattern=True, tolerance=0.001):
    """Checks if two trendlines are diverging."""
    if is_rising_pattern: # Both rising, lower steeper (e.g., ascending channel)
        return slope1 > 0 and slope2 > 0 and slope2 > slope1
    else: # Both falling, upper steeper (e.g., descending channel)
        return slope1 < 0 and slope2 < 0 and abs(slope1) > abs(slope2)


# --- Chart Pattern Detection Functions ---
# (These functions are simplified for illustrative purposes and would require more robust
# validation in a production environment)

def detect_double_top_bottom(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, price_tolerance: float = 0.01) -> dict:
    """Detects Double Top and Double Bottom patterns."""
    patterns = {'has_double_top': False, 'has_double_bottom': False}
    
    if len(df) < 50: return patterns

    close_prices = df['close']
    
    # Double Top: Two peaks at similar levels, separated by a trough.
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            p1_idx_arr = peaks[i]
            p2_idx_arr = peaks[i+1]
            
            # Ensure indices are valid for iloc
            if p1_idx_arr >= len(close_prices) or p2_idx_arr >= len(close_prices):
                continue

            p1_price = close_prices.iloc[p1_idx_arr]
            p2_price = close_prices.iloc[p2_idx_arr]

            if p2_idx_arr - p1_idx_arr > 5: # Ensure sufficient separation
                # Find trough between p1 and p2 using original df indices
                troughs_in_between = [t for t in troughs if p1_idx_arr < t < p2_idx_arr]
                if troughs_in_between:
                    min_price_in_between = close_prices.iloc[troughs_in_between].min()
                    
                    if (abs(p1_price - p2_price) / p1_price < price_tolerance and # Peaks are similar
                        min_price_in_between < p1_price * 0.95): # Trough is significantly lower (5% drop)
                        patterns['has_double_top'] = True
                        if VERBOSE_MODE: logging.info(f"  Double Top detected!")
                        break # Found one, exit loop

    # Double Bottom: Two troughs at similar levels, separated by a peak.
    if len(troughs) >= 2:
        for i in range(len(troughs) - 1):
            t1_idx_arr = troughs[i]
            t2_idx_arr = troughs[i+1]

            if t1_idx_arr >= len(close_prices) or t2_idx_arr >= len(close_prices):
                continue
            
            t1_price = close_prices.iloc[t1_idx_arr]
            t2_price = close_prices.iloc[t2_idx_arr]

            if t2_idx_arr - t1_idx_arr > 5: # Ensure sufficient separation
                # Find peak between t1 and t2
                peaks_in_between = [p for p in peaks if t1_idx_arr < p < t2_idx_arr]
                if peaks_in_between:
                    max_price_in_between = close_prices.iloc[peaks_in_between].max()
                    if (abs(t1_price - t2_price) / t1_price < price_tolerance and # Troughs are similar
                        max_price_in_between > t1_price * 1.05): # Peak is significantly higher (5% rise)
                        patterns['has_double_bottom'] = True
                        if VERBOSE_MODE: logging.info(f"  Double Bottom detected!")
                        break # Found one, exit loop

    return patterns

def detect_triple_top_bottom(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, price_tolerance: float = 0.015) -> dict:
    """Detects Triple Top and Triple Bottom patterns."""
    patterns = {'has_triple_top': False, 'has_triple_bottom': False}
    if len(df) < 75: return patterns # Needs more data for triple patterns

    close_prices = df['close']

    # Triple Top: Three peaks at similar levels
    if len(peaks) >= 3:
        for i in range(len(peaks) - 2):
            p1_idx_arr, p2_idx_arr, p3_idx_arr = peaks[i], peaks[i+1], peaks[i+2]
            
            if p3_idx_arr >= len(close_prices): continue

            p1_price, p2_price, p3_price = close_prices.iloc[p1_idx_arr], close_prices.iloc[p2_idx_arr], close_prices.iloc[p3_idx_arr]

            if (p2_idx_arr - p1_idx_arr > 5 and p3_idx_arr - p2_idx_arr > 5 and # Sufficient separation
                abs(p1_price - p2_price) / p1_price < price_tolerance and
                abs(p2_price - p3_price) / p3_price < price_tolerance): # Peaks are similar (corrected comparison with p3_price)
                # Check for two troughs between them
                troughs_in_between = [t for t in troughs if t > p1_idx_arr and t < p3_idx_arr]
                if len(troughs_in_between) >= 2:
                    patterns['has_triple_top'] = True
                    if VERBOSE_MODE: logging.info(f"  Triple Top detected!")
                    break

    # Triple Bottom: Three troughs at similar levels
    if len(troughs) >= 3:
        for i in range(len(troughs) - 2):
            t1_idx_arr, t2_idx_arr, t3_idx_arr = troughs[i], troughs[i+1], troughs[i+2]
            
            if t3_idx_arr >= len(close_prices): continue

            t1_price, t2_price, t3_price = close_prices.iloc[t1_idx_arr], close_prices.iloc[t2_idx_arr], close_prices.iloc[t3_idx_arr]

            if (t2_idx_arr - t1_idx_arr > 5 and t3_idx_arr - t2_idx_arr > 5 and # Sufficient separation
                abs(t1_price - t2_price) / t1_price < price_tolerance and
                abs(t2_price - t3_price) / t3_price < price_tolerance): # Troughs are similar (corrected comparison with t3_price)
                # Check for two peaks between them
                peaks_in_between = [p for p in peaks if p > t1_idx_arr and p < t3_idx_arr]
                if len(peaks_in_between) >= 2:
                    patterns['has_triple_bottom'] = True
                    if VERBOSE_MODE: logging.info(f"  Triple Bottom detected!")
                    break

    return patterns

def detect_head_and_shoulders_patterns(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> dict:
    """Detects Head and Shoulders (H&S) and Inverse H&S patterns."""
    patterns = {'has_head_and_shoulders': False, 'has_inverse_head_and_shoulders': False}
    if len(df) < 100: return patterns

    close_prices = df['close']
    
    # Head and Shoulders (Bearish)
    if len(peaks) >= 3 and len(troughs) >= 2:
        for i in range(len(peaks) - 2):
            ls_idx, h_idx, rs_idx = peaks[i], peaks[i+1], peaks[i+2]
            
            if rs_idx >= len(close_prices): continue

            ls_price, h_price, rs_price = close_prices.iloc[ls_idx], close_prices.iloc[h_idx], close_prices.iloc[rs_idx]

            # Head is highest, shoulders are lower and roughly equal in height
            if (h_price > ls_price and h_price > rs_price and
                abs(ls_price - rs_price) / ls_price < 0.03 and # Shoulders are similar (within 3%)
                h_price > ls_price * 1.05 and h_price > rs_price * 1.05 and # Head significantly higher (5%)
                ls_idx < h_idx < rs_idx):
                
                # Check for neckline troughs
                trough1_candidates = [t for t in troughs if ls_idx < t < h_idx]
                trough2_candidates = [t for t in troughs if h_idx < t < rs_idx]
                
                if trough1_candidates and trough2_candidates:
                    neckline_trough1_idx = trough1_candidates[-1] # Take the most recent one
                    neckline_trough2_idx = trough2_candidates[0] # Take the earliest one

                    # Ensure neckline is relatively flat
                    slope, _, r_squared = fit_trendline(close_prices, np.array([neckline_trough1_idx, neckline_trough2_idx]))
                    if is_flat(slope, tolerance=0.005) and r_squared > 0.5:
                        patterns['has_head_and_shoulders'] = True
                        if VERBOSE_MODE: logging.info(f"  Head and Shoulders detected!")
                        break

    # Inverse Head and Shoulders (Bullish)
    if len(troughs) >= 3 and len(peaks) >= 2:
        for i in range(len(troughs) - 2):
            ls_idx, h_idx, rs_idx = troughs[i], troughs[i+1], troughs[i+2]
            
            if rs_idx >= len(close_prices): continue

            ls_price, h_price, rs_price = close_prices.iloc[ls_idx], close_prices.iloc[h_idx], close_prices.iloc[rs_idx]

            # Head is lowest, shoulders are higher and roughly equal in depth
            if (h_price < ls_price and h_price < rs_price and
                abs(ls_price - rs_price) / ls_price < 0.03 and # Shoulders are similar (within 3%)
                h_price < ls_price * 0.95 and h_price < rs_price * 0.95 and # Head significantly lower (5%)
                ls_idx < h_idx < rs_idx):
                
                # Check for neckline peaks
                peak1_candidates = [p for p in peaks if ls_idx < p < h_idx]
                peak2_candidates = [p for p in peaks if h_idx < p < rs_idx]

                if peak1_candidates and peak2_candidates:
                    neckline_peak1_idx = peak1_candidates[-1]
                    neckline_peak2_idx = peak2_candidates[0]

                    # Ensure neckline is relatively flat
                    slope, _, r_squared = fit_trendline(close_prices, np.array([neckline_peak1_idx, neckline_peak2_idx]))
                    if is_flat(slope, tolerance=0.005) and r_squared > 0.5:
                        patterns['has_inverse_head_and_shoulders'] = True
                        if VERBOSE_MODE: logging.info(f"  Inverse Head and Shoulders detected!")
                        break

    return patterns

def detect_wedge_patterns(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, lookback_window: int = 50) -> dict:
    """Detects Rising and Falling Wedge patterns."""
    patterns = {'has_rising_wedge': False, 'has_falling_wedge': False}
    if len(df) < lookback_window: return patterns

    close_prices = df['close']
    
    # Consider only recent peaks/troughs
    recent_peaks = sorted([p for p in peaks if p >= len(df) - lookback_window])
    recent_troughs = sorted([t for t in troughs if t >= len(df) - lookback_window])

    if len(recent_peaks) < 2 or len(recent_troughs) < 2:
        return patterns

    upper_slope, _, upper_r_sq = fit_trendline(close_prices, np.array(recent_peaks))
    lower_slope, _, lower_r_sq = fit_trendline(close_prices, np.array(recent_troughs))

    if upper_r_sq > 0.6 and lower_r_sq > 0.6: # Good fit for trendlines
        if upper_slope > 0 and lower_slope > 0 and upper_slope > lower_slope * 1.2 and is_converging(upper_slope, lower_slope, True):
            patterns['has_rising_wedge'] = True
            if VERBOSE_MODE: logging.info(f"  Rising Wedge detected!")

    # Falling Wedge (Bullish): converging downward trendlines (support steeper than resistance)
    if upper_r_sq > 0.6 and lower_r_sq > 0.6:
        if upper_slope < 0 and lower_slope < 0 and abs(lower_slope) > abs(upper_slope) * 1.2 and is_converging(upper_slope, lower_slope, False):
            patterns['has_falling_wedge'] = True
            if VERBOSE_MODE: logging.info(f"  Falling Wedge detected!")
            
    return patterns

def detect_triangle_patterns(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, lookback_window: int = 50) -> dict:
    """Detects Ascending, Descending, and Symmetrical Triangle patterns."""
    patterns = {'has_ascending_triangle': False, 'has_descending_triangle': False, 'has_symmetrical_triangle': False}
    if len(df) < lookback_window: return patterns

    close_prices = df['close']
    
    recent_peaks = sorted([p for p in peaks if p >= len(df) - lookback_window])
    recent_troughs = sorted([t for t in troughs if t >= len(df) - lookback_window])

    if len(recent_peaks) < 2 or len(recent_troughs) < 2:
        return patterns

    upper_slope, _, upper_r_sq = fit_trendline(close_prices, np.array(recent_peaks))
    lower_slope, _, lower_r_sq = fit_trendline(close_prices, np.array(recent_troughs))

    if upper_r_sq > 0.6 and lower_r_sq > 0.6: # Good fit for trendlines
        # Ascending Triangle (Bullish): Flat resistance, rising support
        if is_flat(upper_slope, tolerance=0.0005) and lower_slope > 0.0005:
            patterns['has_ascending_triangle'] = True
            if VERBOSE_MODE: logging.info(f"  Ascending Triangle detected!")

        # Descending Triangle (Bearish): Flat support, falling resistance
        if is_flat(lower_slope, tolerance=0.0005) and upper_slope < -0.0005:
            patterns['has_descending_triangle'] = True
            if VERBOSE_MODE: logging.info(f"  Descending Triangle detected!")

        # Symmetrical Triangle: Converging trendlines (resistance falling, support rising)
        if upper_slope < -0.0005 and lower_slope > 0.0005:
            patterns['has_symmetrical_triangle'] = True
            if VERBOSE_MODE: logging.info(f"  Symmetrical Triangle detected!")

    return patterns

def detect_flag_patterns(df: pd.DataFrame, min_pole_len: int = 10, min_flag_len: int = 5, pole_change_threshold: float = 0.05, consolidation_pct: float = 0.03) -> dict:
    """
    Basic detection for Bullish and Bearish Flags.
    Looks for a strong move (pole) followed by a short, counter-trend consolidation (flag).
    """
    patterns = {'has_bullish_flag': False, 'has_bearish_flag': False}
    
    # Ensure enough data for at least a minimal pole and flag
    if len(df) < (min_pole_len + min_flag_len + 1): # +1 because last element is at index len-1
        return patterns

    close_prices = df['close']
    
    # Iterate through possible starting points (i) for the pole.
    # The pole end will be at `i + min_pole_len`. This index must be valid.
    # The flag starts after the pole and needs `min_flag_len` elements.
    # So the latest possible start for `i` is when `i + min_pole_len + min_flag_len` equals `len(close_prices) - 1`.
    # Therefore, `i` can go up to `len(close_prices) - min_pole_len - min_flag_len - 1`.
    # The `range` function is exclusive of the stop value, so we add 1.
    # We start searching from 0 (the beginning of the recent_df_for_patterns slice).
    max_i_for_pole_start = len(close_prices) - min_pole_len - min_flag_len
    
    # Ensure there's at least one valid starting point for the pole
    if max_i_for_pole_start < 0:
        return patterns

    for i in range(max_i_for_pole_start + 1):
        pole_start_price = close_prices.iloc[i]
        pole_end_price = close_prices.iloc[i + min_pole_len] # This is now guaranteed to be in bounds

        # Bullish Flag
        if (pole_end_price - pole_start_price) / pole_start_price > pole_change_threshold:
            # Found a potential pole, now check for flag consolidation
            flag_slice = close_prices.iloc[i + min_pole_len : ]
            if len(flag_slice) >= min_flag_len:
                flag_max = flag_slice.max()
                flag_min = flag_slice.min()
                # Flag should be a tight, typically downward-sloping (counter-trend) consolidation
                if (flag_max - flag_min) / pole_end_price < consolidation_pct: # Narrow range
                    # Check if it's counter-trend (i.e., slightly downward)
                    flag_slope, _, _ = fit_trendline(flag_slice, np.arange(len(flag_slice)))
                    if flag_slope < 0:
                        patterns['has_bullish_flag'] = True
                        if VERBOSE_MODE: logging.info(f"  Bullish Flag detected!")
                        # No `break` here, allow other patterns to be detected if multiple exist, but usually flags are single occurrences per scan.
                        # For simplicity, if we detect one, we can stop early, but leaving it to continue might find other valid ones.
                        # For now, will allow it to continue to potentially find other flags in the same window.
                        break # Found one, exit loop for this pattern

        # Bearish Flag (similar logic, strong downward pole, slight upward flag)
        if (pole_end_price - pole_start_price) / pole_start_price < -pole_change_threshold:
            # Found a potential pole, now check for flag consolidation
            flag_slice = close_prices.iloc[i + min_pole_len : ]
            if len(flag_slice) >= min_flag_len:
                flag_max = flag_slice.max()
                flag_min = flag_slice.min()
                # Flag should be a tight, typically upward-sloping (counter-trend) consolidation
                if (flag_max - flag_min) / pole_end_price < consolidation_pct: # Narrow range
                    # Check if it's counter-trend (i.e., slightly upward)
                    flag_slope, _, _ = fit_trendline(flag_slice, np.arange(len(flag_slice)))
                    if flag_slope > 0:
                        patterns['has_bearish_flag'] = True
                        if VERBOSE_MODE: logging.info(f"  Bearish Flag detected!")
                        break # Found one, exit loop for this pattern

    return patterns

def detect_cup_and_handle_pattern(df: pd.DataFrame, lookback_window: int = 60, min_cup_depth_pct: float = 0.10, max_handle_retrace_pct: float = 0.5) -> dict:
    """
    Basic detection for Cup and Handle pattern (bullish continuation).
    Looks for a 'U' shape (cup) followed by a short downward drift (handle).
    This is a simplification and would require more robust peak/trough analysis for production.
    """
    patterns = {'has_cup_and_handle': False}
    if len(df) < lookback_window: return patterns

    close_prices = df['close']
    
    # Analyze the most recent data
    recent_df = df.iloc[-lookback_window:].copy()
    recent_close = recent_df['close']
    
    # Find potential left lip (LL) and right lip (RL) of the cup
    # The cup should be a U-shape where LL and RL are high points, and the middle is a low point
    # This simplified detection looks for a period of decline, then a recovery to near the start.
    
    # Find the deepest trough in the first 70% of the lookback window (potential cup bottom)
    cup_trough_idx = recent_close.iloc[:int(len(recent_close)*0.7)].idxmin()
    cup_trough_price = recent_close.loc[cup_trough_idx]

    # The 'left lip' is assumed to be near the start of the window
    left_lip_price = recent_close.iloc[0]

    # The 'right lip' should be a peak formed after the cup trough, near the level of the left lip
    # Look for the peak after the cup trough that is closest to left_lip_price
    potential_right_lip_slice = recent_close.loc[cup_trough_idx:]
    if potential_right_lip_slice.empty: return patterns

    potential_right_lip_idx = potential_right_lip_slice.idxmax()
    right_lip_price = recent_close.loc[potential_right_lip_idx]

    # Check cup shape:
    # 1. Left and Right lips are roughly at the same level
    # 2. Cup depth is significant
    if (abs(left_lip_price - right_lip_price) / left_lip_price < 0.05 and # Lips similar (within 5%)
        (left_lip_price - cup_trough_price) / left_lip_price > min_cup_depth_pct): # Sufficient cup depth (e.g., 10%)

        # Now, check for the handle. The handle should be a small downward retrace
        # after the right lip, staying above the cup's depth.
        handle_start_idx = recent_df.index.get_loc(potential_right_lip_idx)
        if handle_start_idx + 5 >= len(recent_df): # Need at least 5 bars for a handle
            return patterns

        handle_slice = recent_df.iloc[handle_start_idx:].copy()
        
        handle_high = handle_slice['high'].max()
        handle_low = handle_slice['low'].min()
        
        # Handle retrace should not go too deep into the cup
        # (right_lip_price - handle_low) / (right_lip_price - cup_trough_price) < max_handle_retrace_pct
        
        # Simpler check: handle should be a small downward move from the right lip
        if handle_slice['close'].iloc[-1] < handle_slice['close'].iloc[0] and \
           (handle_high - handle_low) / right_lip_price < 0.05: # Handle is tight and downward
            patterns['has_cup_and_handle'] = True
            if VERBOSE_MODE: logging.info(f"  Cup and Handle detected!")

    return patterns

# --- Main Data & Feature Processing Functions ---

def fetch_data(ticker: str, period: str, interval: str, alpha_vantage_api_key: str) -> tuple[pd.DataFrame | None, dict, dict]:
    """
    Fetches historical OHLCV data from yfinance and current fundamental/news data from Alpha Vantage.
    Returns: df (DataFrame), fundamental_data (dict), news_sentiment (dict)
    """
    logging.info(f"  [Data Fetch] Fetching data for {ticker} (yfinance)...")
    # Set auto_adjust to True to avoid Adjusted Close column issues
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        logging.warning(f"  [Data Fetch] No data returned for {ticker} from yfinance for period='{period}', interval='{interval}'.")
        return None, {}, {}

    # Handle multi-level columns from yfinance, which can occur with certain settings.
    # If a column is a tuple, take its first element.
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            cleaned_col = str(col[0]).replace(' ', '_').lower()
        else:
            cleaned_col = str(col).replace(' ', '_').lower()
        new_columns.append(cleaned_col)
    df.columns = new_columns

    # Ensure essential columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"  [Data Fetch Error] Missing one or more essential OHLCV columns for {ticker}. Found: {df.columns.tolist()}")
        return None, {}, {}
    
    # Handle missing volume if any
    if 'volume' not in df.columns or df['volume'].isnull().all():
        df['volume'] = 0 # Default to 0 volume if missing

    # Ensure all OHLCV columns are numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Drop rows with missing critical OHLC

    if df.empty:
        logging.warning(f"  [Data Fetch] DataFrame is empty after cleaning OHLC NaNs for {ticker}.")
        return None, {}, {}

    fundamental_data = {}
    news_sentiment_data = {}

    if alpha_vantage_api_key and alpha_vantage_api_key != "YOUR_ALPHA_VANTAGE_API_KEY":
        try:
            # 1. Fetch Company Overview for fundamental ratios
            logging.info(f"  [Data Fetch] Fetching fundamental data for {ticker} (Alpha Vantage OVERVIEW)...")
            overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker.split('.')[0]}&apikey={alpha_vantage_api_key}"
            overview_response = requests.get(overview_url, timeout=10)
            overview_response.raise_for_status()
            overview_data = overview_response.json()

            if overview_data and not overview_data.get('Error Message'):
                fundamental_data['pe_ratio'] = float(overview_data.get('PERatio', np.nan)) if overview_data.get('PERatio', 'None') != 'None' else np.nan
                fundamental_data['market_capitalization'] = float(overview_data.get('MarketCapitalization', np.nan)) if overview_data.get('MarketCapitalization', 'None') != 'None' else np.nan
                fundamental_data['book_value_per_share'] = float(overview_data.get('BookValuePerShare', np.nan)) if overview_data.get('BookValuePerShare', 'None') != 'None' else np.nan
                fundamental_data['dividend_yield'] = float(overview_data.get('DividendYield', np.nan)) if overview_data.get('DividendYield', 'None') != 'None' else np.nan
                logging.info(f"  [Data Fetch] Fundamental data for {ticker}: {fundamental_data}")
            else:
                logging.warning(f"  [Data Fetch] No fundamental data or error for {ticker} from Alpha Vantage Overview: {overview_data.get('Error Message', 'Unknown')}")

            # 2. Fetch News Sentiment
            logging.info(f"  [Data Fetch] Fetching news sentiment for {ticker} (Alpha Vantage NEWS_SENTIMENT)...")
            # Alpha Vantage News Sentiment API takes only the base ticker symbol
            news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker.split('.')[0]}&sort=LATEST&limit=5&apikey={alpha_vantage_api_key}"
            news_response = requests.get(news_url, timeout=10)
            news_response.raise_for_status()
            news_data = news_response.json()

            total_sentiment_score = 0
            sentiment_count = 0
            if 'feed' in news_data and news_data['feed']:
                for article in news_data['feed']:
                    if 'overall_sentiment_score' in article and article['overall_sentiment_score'] is not None:
                        try:
                            total_sentiment_score += float(article['overall_sentiment_score'])
                            sentiment_count += 1
                        except ValueError:
                            logging.warning(f"  [Data Fetch] Could not parse sentiment score: {article['overall_sentiment_score']}")
                if sentiment_count > 0:
                    news_sentiment_data['avg_news_sentiment'] = total_sentiment_score / sentiment_count
                else:
                    news_sentiment_data['avg_news_sentiment'] = 0.0 # Default to neutral if no valid scores
            else:
                news_sentiment_data['avg_news_sentiment'] = 0.0 # Default to neutral if no feed
            logging.info(f"  [Data Fetch] News sentiment for {ticker}: {news_sentiment_data}")

        except requests.exceptions.Timeout:
            logging.error(f"  [Data Fetch Error] Alpha Vantage API request timed out for {ticker}.")
        except requests.exceptions.ConnectionError:
            logging.error(f"  [Data Fetch Error] Network connection error while fetching Alpha Vantage data for {ticker}.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"  [Data Fetch Error] HTTP error fetching Alpha Vantage data for {ticker}: {e}. Check API key and rate limits.")
        except json.JSONDecodeError:
            logging.error(f"  [Data Fetch Error] Error decoding JSON from Alpha Vantage for {ticker}. Response: {news_response.text if 'news_response' in locals() else overview_response.text}")
        except Exception as e:
            logging.error(f"  [Data Fetch Error] An unexpected error occurred with Alpha Vantage for {ticker}: {e}")
    else:
        logging.warning("  [Data Fetch] Alpha Vantage API Key not set or is a placeholder. Skipping fundamental and news data fetching.")

    return df, fundamental_data, news_sentiment_data

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of technical indicators using pandas_ta.
    Each indicator call is wrapped in a try-except block for robustness.
    """
    if df is None or df.empty:
        logging.warning("[Indicator Calc] Input DataFrame is empty or None. Skipping indicator calculation.")
        return df

    df_copy = df.copy()

    # Ensure OHLCV columns are numeric for pandas_ta
    required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    for col in required_ohlcv:
        if col not in df_copy.columns:
            logging.warning(f"  [Indicator Calc Warning] Missing critical OHLCV column '{col}'. Some indicators may fail.")
            return df_copy # Cannot proceed if essential columns are truly missing
        
        # Ensure column is numeric
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy[col].fillna(method='ffill', inplace=True)
        df_copy[col].fillna(method='bfill', inplace=True)
        df_copy[col].fillna(0, inplace=True) # Final fallback for NaNs

    def _safe_ta_call(df_target, indicator_name_str, **kwargs):
        """Helper to safely call pandas_ta indicators."""
        try:
            # Dynamically get the indicator function from pandas_ta.
            # This avoids passing 'df_target' twice, which caused the "multiple values" error.
            indicator_func = getattr(df_target.ta, indicator_name_str)
            indicator_func(**kwargs, append=True)
        except AttributeError:
            logging.warning(f"  [Indicator Calc Warning] Indicator '{indicator_name_str}' not found or available in pandas_ta for this version/setup. Skipping.")
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
        # Removed the direct call for chv to let _safe_ta_call handle potential AttributeErrors
        _safe_ta_call(df_copy, 'chv') # Chaikin Money Flow - now uses safe call
        _safe_ta_call(df_copy, 'vwap') # VWAP
    else:
        if VERBOSE_MODE: logging.warning(f"  [Indicator Calc Warning] Skipping volume-based indicators (VWAP, MFI, etc.) due to lack of meaningful volume data.")

    # --- Other / Composite Indicators ---
    _safe_ta_call(df_copy, 'psar') # Parabolic SAR (often PSARl, PSARs, PSARaf)
    _safe_ta_call(df_copy, 'supertrend') # Supertrend (SUPERTd, SUPERTl, SUPERTs)
    _safe_ta_call(df_copy, 'fisher') # Fisher Transform (FISHERT, FISHERTs)
    _safe_ta_call(df_copy, 'rvi') # Relative Vigor Index

    # Clean up column names (replace non-alphanumeric chars)
    df_copy.columns = [str(col).replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '') for col in df_copy.columns]
    
    # Fill NaNs created by indicators. Prioritize ffill/bfill then a final 0.
    for col in df_copy.columns:
        if df_copy[col].dtype in ['float64', 'int64']: # Only apply to numeric columns
            df_copy[col].replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf/-inf with NaN
            df_copy[col].fillna(method='ffill', inplace=True)
            df_copy[col].fillna(method='bfill', inplace=True)
            if df_copy[col].isnull().any(): # If NaNs still remain (e.g., at beginning of series)
                df_copy[col].fillna(0, inplace=True) # Fill remaining with 0 or a sensible value

    logging.info(f"[Indicator Calc] Indicators calculated. Final shape: {df_copy.shape}")
    nan_counts = df_copy.isnull().sum()
    if nan_counts.sum() > 0:
        logging.warning(f"[Indicator Calc WARNING] NaNs still present in columns after filling: \n{nan_counts[nan_counts > 0].to_string()}")

    return df_copy


def detect_all_chart_patterns(df: pd.DataFrame, min_required_bars: int) -> dict:
    """
    Detects a comprehensive set of chart patterns and returns them as a dictionary of boolean flags.
    """
    pattern_flags = {
        'has_double_top': False, 'has_double_bottom': False,
        'has_triple_top': False, 'has_triple_bottom': False,
        'has_head_and_shoulders': False, 'has_inverse_head_and_shoulders': False,
        'has_rising_wedge': False, 'has_falling_wedge': False,
        'has_ascending_triangle': False, 'has_descending_triangle': False, 'has_symmetrical_triangle': False,
        'has_bullish_flag': False, 'has_bearish_flag': False,
        'has_cup_and_handle': False,
    }

    if df.empty or 'close' not in df.columns or len(df) < min_required_bars:
        if VERBOSE_MODE:
            logging.warning(f"[Pattern Detection] Insufficient data ({len(df)} bars) for comprehensive pattern detection. Min required: {min_required_bars}.")
        return pattern_flags

    # Use a dynamic window for pattern detection, typically 50-100 bars for daily data
    # or proportionally smaller for intraday if relevant
    pattern_detection_window = min(len(df), 100)
    recent_df_for_patterns = df.iloc[-pattern_detection_window:].copy()

    # Re-calculate peaks and troughs for the relevant window
    peaks, troughs = find_peaks_and_troughs(recent_df_for_patterns['close'], order=5) # Order 5 is reasonable for daily

    if recent_df_for_patterns.empty or len(peaks) < 2 or len(troughs) < 2:
        return pattern_flags # Not enough distinct points for pattern detection

    # IMPORTANT: The pattern detection functions below typically expect array indices
    # relative to the `recent_df_for_patterns` subset.
    # The `find_peaks_and_troughs` function already returns array indices.

    # Double/Triple Tops/Bottoms
    dt_db_patterns = detect_double_top_bottom(recent_df_for_patterns, peaks, troughs)
    pattern_flags.update(dt_db_patterns)

    tt_tb_patterns = detect_triple_top_bottom(recent_df_for_patterns, peaks, troughs)
    pattern_flags.update(tt_tb_patterns)

    # Head and Shoulders
    hs_ihs_patterns = detect_head_and_shoulders_patterns(recent_df_for_patterns, peaks, troughs)
    pattern_flags.update(hs_ihs_patterns)

    # Wedges
    wedge_patterns = detect_wedge_patterns(recent_df_for_patterns, peaks, troughs)
    pattern_flags.update(wedge_patterns)

    # Triangles
    triangle_patterns = detect_triangle_patterns(recent_df_for_patterns, peaks, troughs)
    pattern_flags.update(triangle_patterns)

    # Flags
    flag_patterns = detect_flag_patterns(recent_df_for_patterns)
    pattern_flags.update(flag_patterns)

    # Cup and Handle
    cup_handle_pattern = detect_cup_and_handle_pattern(recent_df_for_patterns)
    pattern_flags.update(cup_handle_pattern)
    
    if VERBOSE_MODE:
        detected_true_patterns = {k: v for k, v in pattern_flags.items() if v}
        if detected_true_patterns:
            logging.info(f"[Pattern Detection] Detected patterns for latest data: {detected_true_patterns}")
        else:
            logging.info("[Pattern Detection] No significant patterns detected.")

    return pattern_flags


def create_target_variable(df: pd.DataFrame, future_periods: int, up_threshold: float, down_threshold: float) -> pd.DataFrame | None:
    """
    Creates a categorical target variable (Buy, Sell, Hold) based on future price movements.
    """
    if df is None or df.empty:
        if VERBOSE_MODE:
            logging.warning("[Target Creation] Input DataFrame is empty or None. Cannot create target.")
        return None

    if 'close' not in df.columns or not pd.api.types.is_numeric_dtype(df['close']):
        if VERBOSE_MODE:
            logging.warning("[Target Creation Warning] 'close' column is missing or not numeric. Cannot create target variable.")
        return None

    df_copy = df.copy()
    
    # Calculate future close price
    df_copy['future_close'] = df_copy['close'].shift(-future_periods)
    # Calculate percentage price change
    df_copy['price_change'] = (df_copy['future_close'] - df_copy['close']) / df_copy['close']

    # Define target: 1 for Buy, 0 for Sell, 2 for Hold
    df_copy['target'] = 2 # Default to Hold
    df_copy.loc[df_copy['price_change'] >= up_threshold, 'target'] = 1 # Buy
    df_copy.loc[df_copy['price_change'] <= down_threshold, 'target'] = 0 # Sell

    # Drop columns used for target creation
    df_copy.drop(columns=['future_close', 'price_change'], inplace=True)
    
    # Drop rows where target could not be calculated (at the end of the DataFrame)
    initial_rows = df_copy.shape[0]
    df_copy.dropna(subset=['target'], inplace=True)
    if VERBOSE_MODE and (initial_rows - df_copy.shape[0]) > 0:
        logging.info(f"[Target Creation] Dropped {initial_rows - df_copy.shape[0]} rows due to NaN in target variable (last {future_periods} rows).")

    # Ensure target is integer type
    df_copy['target'] = df_copy['target'].astype(int)

    if VERBOSE_MODE:
        logging.info(f"[Target Creation] Label distribution (0=Sell, 1=Buy, 2=Hold):\n{df_copy['target'].value_counts()}")
        logging.info(f"[Target Creation] Shape after target creation: {df_copy.shape}")
    return df_copy

def prepare_features_for_ai(df: pd.DataFrame, fundamental_data: dict = None, news_sentiment: dict = None, for_training: bool = True, known_features: list = None) -> tuple[pd.DataFrame, pd.Series | None, list | None]:
    """
    Prepares the DataFrame for AI model by selecting features and handling NaNs.
    Adds fundamental data, news sentiment, and binary pattern features.
    """
    logging.info("[Feature Prep] Preparing features for AI model...")

    df_processed = df.copy()

    # --- Add Fundamental Data and News Sentiment as features ---
    # For training: these columns will be mostly NaN historically unless a historical source is used.
    # For prediction: these will be current values for the single row.
    if fundamental_data:
        for key, value in fundamental_data.items():
            df_processed[key] = value # This will broadcast
    if news_sentiment:
        for key, value in news_sentiment.items():
            df_processed[key] = value # This will broadcast

    # --- Add Chart Pattern features (binary 0/1) ---
    pattern_feature_names = [
        'has_double_top', 'has_double_bottom', 'has_triple_top', 'has_triple_bottom',
        'has_head_and_shoulders', 'has_inverse_head_and_shoulders',
        'has_rising_wedge', 'has_falling_wedge',
        'has_ascending_triangle', 'has_descending_triangle', 'has_symmetrical_triangle',
        'has_bullish_flag', 'has_bearish_flag', 'has_cup_and_handle'
    ]
    for pattern_name in pattern_feature_names:
        if pattern_name not in df_processed.columns: # Add as 0 if not already present
            df_processed[pattern_name] = 0


    # Exclude non-feature columns (these should never be features)
    excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'target'] 
    
    X = pd.DataFrame() # Initialize X
    y = None
    final_features_list = None

    if for_training:
        # Start with all numeric columns as potential features
        all_potential_features = [col for col in df_processed.columns if df_processed[col].dtype in ['float64', 'int64'] and col not in excluded_cols]

        # Initialize features_to_use with all features that are ALWAYS_INCLUDED_FEATURES
        features_to_use = [f for f in ALWAYS_INCLUDED_FEATURES if f in df_processed.columns]

        # Add other potential features ONLY if they have variance and are not all NaN
        for f in all_potential_features:
            if f not in features_to_use: # Avoid adding duplicates
                if df_processed[f].nunique() > 1 and df_processed[f].notna().any():
                    features_to_use.append(f)
        
        if VERBOSE_MODE and len(features_to_use) < len(all_potential_features):
            removed_features = set(all_potential_features) - set(features_to_use)
            logging.info(f"  [Feature Prep] Removed {len(removed_features)} features (not in ALWAYS_INCLUDED_FEATURES) that were constant, single-valued, or all NaN: {list(removed_features)}")

        if not features_to_use:
            logging.error("[Feature Prep] No valid features remaining after filtering for training.")
            return pd.DataFrame(), None, None

        X = df_processed[features_to_use]
        y = df_processed['target'] if 'target' in df_processed.columns else None
        final_features_list = features_to_use

        # Impute any remaining NaNs for training data
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    col_mean = X[col].mean()
                    if pd.isna(col_mean): # If the entire column is NaN, fill with 0
                        X[col].fillna(0, inplace=True)
                    else:
                        X[col].fillna(col_mean, inplace=True)
                else: 
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
        
        # Drop rows where target is NaN (should be handled by create_target_variable but double check)
        if y is not None:
            combined_df = X.copy()
            combined_df['target'] = y
            combined_df.dropna(subset=['target'], inplace=True)
            X = combined_df[features_to_use]
            y = combined_df['target']
            logging.info(f"  [Feature Prep] Shape after target NaN drop: {X.shape}")

    else: # For live prediction (single row)
        if known_features is None:
            logging.error("[Feature Prep] 'known_features' must be provided for prediction mode.")
            return pd.DataFrame(), None, None

        # Ensure that X_pred contains all the 'known_features' from training
        # and in the correct order.
        X_pred_dict = {}
        for feature in known_features:
            if feature in df_processed.columns:
                # For boolean patterns and fundamental/news data, ensure they are numeric (0/1 or actual value)
                value = df_processed[feature].iloc[-1]
                if pd.api.types.is_bool_dtype(df_processed[feature]):
                    X_pred_dict[feature] = int(value)
                elif pd.isna(value):
                    X_pred_dict[feature] = 0 # Impute NaNs in live data with 0
                else:
                    X_pred_dict[feature] = value
            else:
                # This feature was present during training but not found in the current live data.
                X_pred_dict[feature] = 0 # Default to 0 for missing features during prediction
        
        X = pd.DataFrame([X_pred_dict]) # Create a DataFrame from the dictionary
        X.fillna(0, inplace=True) # Ensure no NaNs remain in the prediction row

        final_features_list = known_features # The features used for prediction are exactly the known ones

    logging.info(f"[Feature Prep] Features prepared. Final feature shape: {X.shape}")
    return X, y, final_features_list


def train_and_save_model(df: pd.DataFrame, ticker_name: str) -> tuple[RandomForestClassifier, list] | tuple[None, None]:
    """
    Trains a RandomForestClassifier model using GridSearchCV and saves it along with features.
    Uses TimeSeriesSplit for more realistic cross-validation.
    """
    if df is None or df.empty:
        logging.warning(f"[Model Training] DataFrame is empty or None for {ticker_name}. Cannot train model.")
        return None, None

    # Prepare features for training (including fundamental, news, patterns)
    X, y, features = prepare_features_for_ai(df, for_training=True)

    if X.empty or y is None or len(X) < 10 or y.nunique() < 2:
        logging.warning(f"[Model Training] Not enough valid samples or distinct target labels ({y.nunique() if y is not None else 0}) after feature preparation for meaningful training for {ticker_name}. Training aborted.")
        if VERBOSE_MODE and y is not None:
            logging.info(f"  Target counts: {y.value_counts()}")
        return None, None

    # Use TimeSeriesSplit for more realistic evaluation
    # n_splits=5 means 5 folds. The training set grows, and test set moves forward.
    tscv = TimeSeriesSplit(n_splits=5) 
    
    scorer = make_scorer(f1_score, average='weighted', zero_division=0) # Use weighted F1 for imbalanced classes

    logging.info(f"\n--- Starting GridSearchCV for Hyperparameter Tuning for {ticker_name} ---")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=PARAM_GRID,
        scoring=scorer,
        cv=tscv, # Use TimeSeriesSplit
        verbose=1,
        n_jobs=-1 # Use all available CPU cores
    )

    try:
        grid_search.fit(X, y) # Fit on the entire preprocessed dataset with TimeSeriesSplit
    except Exception as e:
        logging.error(f"[Model Training Error] GridSearchCV failed for {ticker_name}: {e}. Skipping model saving for this ticker.")
        return None, None

    logging.info("\n--- GridSearchCV Results ---")
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    logging.info("Model training complete (Best model from Grid Search selected).")

    # Evaluate on the last fold of TimeSeriesSplit (most recent data used as test)
    # This gives a more realistic "out-of-sample" performance on recent data.
    final_train_idx, final_test_idx = list(tscv.split(X, y))[-1]
    X_train_final, X_test_final = X.iloc[final_train_idx], X.iloc[final_test_idx]
    y_train_final, y_test_final = y.iloc[final_train_idx], y.iloc[final_test_idx]

    if VERBOSE_MODE:
        logging.info(f"\n[Model Training] Final training data shape: {X_train_final.shape}, Final test data shape: {X_test_final.shape}")
        logging.info(f"[Model Training] Final training target distribution:\n{y_train_final.value_counts()}")
        logging.info(f"[Model Training] Final test target distribution:\n{y_test_final.value_counts()}")

    if not X_test_final.empty:
        y_pred = best_model.predict(X_test_final)

        logging.info("\n--- Model Evaluation on Final Test Set (using Best Model) ---")
        logging.info(f"Overall Accuracy: {accuracy_score(y_test_final, y_pred):.4f}")
        logging.info("\nClassification Report (0=Sell, 1=Buy, 2=Hold):")
        target_names_map = {0: 'Sell', 1: 'Buy', 2: 'Hold'}
        
        # Ensure all unique labels from the full 'y' are mapped, even if not in current test set
        all_possible_labels = sorted(y.unique())
        display_target_names = [target_names_map.get(label, f'Label {label}') for label in all_possible_labels]

        logging.info("\n" + classification_report(y_test_final, y_pred, labels=all_possible_labels, target_names=display_target_names, zero_division=0))
        logging.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test_final, y_pred, labels=all_possible_labels))) # Use all_possible_labels for consistent matrix
    else:
        logging.warning("\n--- Model Evaluation: Final test set is empty. No evaluation performed. ---")

    if VERBOSE_MODE:
        logging.info(f"\n--- Top 15 Feature Importances for {ticker_name} (Best Model) ---")
        if hasattr(best_model, 'feature_importances_') and len(features) > 0:
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            with pd.option_context('display.float_format', '{:.6f}'.format):
                logging.info("\n" + feature_importances.head(15).to_string())
        else:
            logging.info("Feature importances not available or no features.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_filename = os.path.join(MODEL_DIR, f"trading_bot_model_{ticker_name.replace('^', '')}.joblib")
    features_filename = os.path.join(MODEL_DIR, f"trading_bot_model_features_{ticker_name.replace('^', '')}.joblib")

    joblib.dump(best_model, model_filename)
    joblib.dump(features, features_filename)
    logging.info(f"\nModel saved to {model_filename}")
    logging.info(f"Features saved to {features_filename}")

    return best_model, features

def load_models_and_features() -> dict:
    """Loads all trained models and their associated features from the MODEL_DIR."""
    loaded_data = {}
    logging.info(f"Loading models from {MODEL_DIR}...")
    for ticker in STOCK_TICKERS:
        clean_ticker_name = ticker.replace('^', '')
        model_path = os.path.join(MODEL_DIR, f"trading_bot_model_{clean_ticker_name}.joblib")
        features_path = os.path.join(MODEL_DIR, f"trading_bot_model_features_{clean_ticker_name}.joblib")

        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                model = joblib.load(model_path)
                features = joblib.load(features_path)
                loaded_data[ticker] = {'model': model, 'features': features}
                if VERBOSE_MODE:
                    logging.info(f"  - Loaded model and features for {ticker}")
            except Exception as e:
                logging.error(f"  - [Model Load Error] Error loading model/features for {ticker}: {e}")
        else:
            logging.warning(f"  - [Model Load Warning] No trained model found for {ticker} at '{model_path}'. Skipping.")
    return loaded_data

def generate_ai_signal(df: pd.DataFrame, model_info: dict, ticker_name: str, fundamental_data: dict, news_sentiment: dict, chart_patterns: dict) -> str:
    """
    Generates an AI trading signal (BUY, SELL, Hold) for the latest data point,
    incorporating fundamental data, news sentiment, and chart patterns.
    """
    model = model_info.get('model')
    known_features = model_info.get('features') # These are the features the model was trained on

    if model is None or known_features is None or not known_features:
        if VERBOSE_MODE:
            logging.warning(f"  [AI Signal {ticker_name}] Model or features not loaded/defined. Cannot predict.")
        return 'Hold (AI Model Error)'

    if df.empty or len(df) == 0: # Ensure df is not empty before proceeding
        if VERBOSE_MODE:
            logging.warning(f"  [AI Signal {ticker_name}] Empty DataFrame for prediction. Cannot generate AI signal.")
        return 'Hold (No Data)'

    # Take the last row of the DataFrame for prediction
    latest_data_row = df.iloc[[-1]].copy() # Ensure it's a DataFrame, not a Series

    # Prepare features for AI prediction, ensuring it matches the training features
    X_pred, _, _ = prepare_features_for_ai(
        latest_data_row,
        fundamental_data=fundamental_data,
        news_sentiment=news_sentiment,
        for_training=False,
        known_features=known_features # Use the exact feature list from training
    )

    if X_pred.empty or X_pred.shape[1] != len(known_features):
        logging.error(f"  [AI Signal {ticker_name}] Mismatch in feature count for prediction. Expected {len(known_features)}, got {X_pred.shape[1]}. Cannot predict.")
        logging.error(f"  [AI Signal {ticker_name}] Missing features: {set(known_features) - set(X_pred.columns)}")
        logging.error(f"  [AI Signal {ticker_name}] Extra features: {set(X_pred.columns) - set(known_features)}")
        return 'Hold (Feature Mismatch)'

    try:
        prediction = model.predict(X_pred)[0]
        # proba = model.predict_proba(X_pred)[0] # If you need confidence scores

        if prediction == 1:
            return 'BUY (AI)'
        elif prediction == 0:
            return 'SELL (AI)'
        else:
            return 'Hold (AI)'
    except Exception as e:
        logging.error(f"  [AI Signal Error {ticker_name}] Unhandled error during AI prediction: {e}")
        return 'Hold (AI Prediction Error)'

def generate_trade_signals(signal: int, current_price: float, fundamental_data: dict, news_sentiment: dict, chart_patterns: dict, recent_high: float, recent_low: float) -> dict:
    """
    Generates actionable trade signals with entry, stop-loss, and take-profit levels.
    These are heuristic-based for demonstration, integrating various data points.
    """
    trade_info = {
        'signal': signal,
        'entry_price': round(current_price, 2),
        'stop_loss': None,
        'take_profit_1': None,
        'take_profit_2': None,
        'rationale': []
    }

    # Add fundamental insights to rationale
    if fundamental_data:
        if fundamental_data.get('pe_ratio') is not None and not pd.isna(fundamental_data['pe_ratio']):
            trade_info['rationale'].append(f"PE Ratio: {fundamental_data['pe_ratio']:.2f}")
        if fundamental_data.get('market_capitalization') is not None and not pd.isna(fundamental_data['market_capitalization']):
            trade_info['rationale'].append(f"Market Cap: {fundamental_data['market_capitalization']/1e9:.2f}B")
        if fundamental_data.get('book_value_per_share') is not None and not pd.isna(fundamental_data['book_value_per_share']):
            trade_info['rationale'].append(f"Book Value/Share: {fundamental_data['book_value_per_share']:.2f}")
        if fundamental_data.get('dividend_yield') is not None and not pd.isna(fundamental_data['dividend_yield']):
            trade_info['rationale'].append(f"Dividend Yield: {fundamental_data['dividend_yield']*100:.2f}%")

    # Add news sentiment to rationale
    if news_sentiment and news_sentiment.get('avg_news_sentiment') is not None and not pd.isna(news_sentiment['avg_news_sentiment']):
        sentiment = news_sentiment['avg_news_sentiment']
        if sentiment > 0.35:
            trade_info['rationale'].append(f"Strong News Sentiment ({sentiment:.2f})")
        elif sentiment < -0.35:
            trade_info['rationale'].append(f"Weak News Sentiment ({sentiment:.2f})")
        else:
            trade_info['rationale'].append(f"Neutral News Sentiment ({sentiment:.2f})")

    # Add chart pattern insights to rationale
    detected_patterns_list = [k for k, v in chart_patterns.items() if v]
    if detected_patterns_list:
        trade_info['rationale'].append(f"Detected Chart Patterns: {', '.join(detected_patterns_list)}")

    # Define stop-loss and take-profit heuristics
    # These are illustrative and should be refined based on risk management strategy
    sl_pct = 0.015 # 1.5% stop loss for general use
    tp1_rr_ratio = 1.5 # Take Profit 1 at 1.5x risk
    tp2_rr_ratio = 3.0 # Take Profit 2 at 3.0x risk

    if signal == 1: # Buy Signal
        trade_info['rationale'].insert(0, "AI BUY SIGNAL") # Prioritize AI signal
        
        # Calculate stop-loss: Use recent low if available and reasonable, else fixed percentage
        if recent_low > 0 and current_price - recent_low > 0:
            # Place SL slightly below recent low, or 1.5% if recent low is too close/far
            trade_info['stop_loss'] = max(recent_low * 0.99, current_price * (1 - sl_pct))
        else:
            trade_info['stop_loss'] = current_price * (1 - sl_pct)

        # Ensure stop_loss is always less than current_price for a buy
        trade_info['stop_loss'] = min(trade_info['stop_loss'], current_price * 0.99) # Add a small buffer

        # Calculate potential profit amount based on risk
        risk_amount = current_price - trade_info['stop_loss']
        if risk_amount <= 0: # Avoid division by zero or negative risk
            risk_amount = current_price * sl_pct # Default to fixed % risk
            trade_info['stop_loss'] = current_price - risk_amount

        trade_info['take_profit_1'] = current_price + (risk_amount * tp1_rr_ratio)
        trade_info['take_profit_2'] = current_price + (risk_amount * tp2_rr_ratio)

        trade_info['rationale'].append(f"Anticipated duration: Short-medium term (based on {LIVE_DATA_INTERVAL} data)")

    elif signal == 0: # Sell Signal
        trade_info['rationale'].insert(0, "AI SELL SIGNAL") # Prioritize AI signal
        
        # Calculate stop-loss: Use recent high if available and reasonable, else fixed percentage
        if recent_high > 0 and recent_high - current_price > 0:
            # Place SL slightly above recent high, or 1.5% if too close/far
            trade_info['stop_loss'] = min(recent_high * 1.01, current_price * (1 + sl_pct))
        else:
            trade_info['stop_loss'] = current_price * (1 + sl_pct)

        # Ensure stop_loss is always greater than current_price for a sell
        trade_info['stop_loss'] = max(trade_info['stop_loss'], current_price * 1.01) # Add a small buffer

        # Calculate potential profit amount for short selling
        risk_amount = trade_info['stop_loss'] - current_price
        if risk_amount <= 0:
            risk_amount = current_price * sl_pct
            trade_info['stop_loss'] = current_price + risk_amount

        trade_info['take_profit_1'] = current_price - (risk_amount * tp1_rr_ratio)
        trade_info['take_profit_2'] = current_price - (risk_amount * tp2_rr_ratio)

        # Ensure take profits for sell are not negative
        trade_info['take_profit_1'] = max(0.01, trade_info['take_profit_1'])
        trade_info['take_profit_2'] = max(0.01, trade_info['take_profit_2'])

        trade_info['rationale'].append(f"Anticipated duration: Short-medium term (based on {LIVE_DATA_INTERVAL} data)")

    else: # Hold Signal (signal == 2)
        trade_info['rationale'].insert(0, "AI HOLD SIGNAL") # Prioritize AI signal
        trade_info['hold_reason'] = "No Strong Signal"
        trade_info['hold_duration'] = "Monitor for next signal (typically next check interval)"

        if chart_patterns.get('has_symmetrical_triangle') or chart_patterns.get('has_rising_wedge') or chart_patterns.get('has_falling_wedge'):
            trade_info['rationale'].append("Market in consolidation/uncertainty. Awaiting breakout.")
            trade_info['hold_reason'] = "Consolidation/Uncertainty"
            trade_info['hold_duration'] = "Until clear trend or pattern breakout"
        
        if news_sentiment and news_sentiment.get('avg_news_sentiment') is not None and abs(news_sentiment['avg_news_sentiment']) < 0.3:
             trade_info['rationale'].append("Neutral news sentiment. No strong directional catalyst.")
             trade_info['hold_reason'] = "Neutral News"
             trade_info['hold_duration'] = "Until significant news"
        
        # Example of how to add more nuanced hold rationales
        if fundamental_data.get('pe_ratio') and fundamental_data['pe_ratio'] > 50:
            trade_info['rationale'].append("High PE Ratio suggests overvaluation (consider reducing position/avoiding buy).")
        elif fundamental_data.get('pe_ratio') and fundamental_data['pe_ratio'] < 10 and current_price > recent_low:
             trade_info['rationale'].append("Low PE Ratio suggests undervaluation, but no immediate technical buy signal.")

    return trade_info

def record_trade(ticker: str, trade_type: str, entry_price: float, exit_price: float = None, signal_time: str = None, trade_details: dict = None):
    """Records a trade action in the global trade log."""
    log_entry = {
        'timestamp': signal_time if signal_time else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'type': trade_type,
        'entry_price': round(entry_price, 2) if entry_price is not None else None,
        'exit_price': round(exit_price, 2) if exit_price is not None else None,
        'p_l_percent': None,
        'details': trade_details # Store full trade details for analysis
    }
    if trade_type == 'EXIT' and exit_price is not None and entry_price is not None and entry_price != 0:
        log_entry['p_l_percent'] = round(((exit_price - entry_price) / entry_price) * 100, 2)
    
    trade_log.append(log_entry)
    if VERBOSE_MODE:
        logging.info(f"  [Trade Log] Recorded: {log_entry}")

def print_trade_log():
    """Prints a summary of trades recorded in the current session."""
    if not trade_log:
        logging.info("\n--- Trade Log (Current Session) ---")
        logging.info("No trades recorded in this session.")
        return

    logging.info("\n--- Trade Log (Current Session) ---")
    df_log = pd.DataFrame(trade_log)
    # Ensure all columns are displayed
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logging.info("\n" + df_log.to_string())

    closed_buys = df_log[df_log['type'] == 'EXIT']
    if not closed_buys.empty:
        total_profit_loss_percent = closed_buys['p_l_percent'].sum()
        logging.info(f"\nTotal P/L for closed positions: {total_profit_loss_percent:.2f}%")
    else:
        logging.info("\nNo closed positions to calculate total P/L.")


# --- Main Monitoring Loop ---

def monitor_markets(loaded_models: dict):
    """
    Continuously monitors markets, fetches data, generates signals, and manages trades.
    """
    logging.info(f"\nStarting real-time market monitor for: {STOCK_TICKERS}")
    logging.info(f"Checking every {CHECK_INTERVAL_SECONDS} seconds...")

    if not loaded_models:
        logging.warning("WARNING: No trained models found. Cannot provide AI-driven signals. Please run in 'train' mode first.")
        logging.info("Exiting monitoring mode as no AI models are available.")
        return

    # Initialize or reset states for a new monitoring session
    for ticker in STOCK_TICKERS:
        if ticker not in stock_states: # Ensure all tickers in config are in state dict
            stock_states[ticker] = {
                'last_ai_signal': 'Hold',
                'current_price': None,
                'in_buy_position': False,
                'entry_price': None,
                'stop_loss_price': None,
                'high_since_entry': None,
                'last_notification_price': None
            }
        # Reset dynamic state for a new session start
        stock_states[ticker]['in_buy_position'] = False 
        stock_states[ticker]['entry_price'] = None
        stock_states[ticker]['stop_loss_price'] = None
        stock_states[ticker]['high_since_entry'] = None
        stock_states[ticker]['last_notification_price'] = None # Forces first notification

    try:
        while True:
            current_check_time = datetime.now()
            timestamp_str = current_check_time.strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"\n--- Checking market at {timestamp_str} ---")

            # Fetch news once per cycle for all tickers (if API key allows)
            # This is already handled inside fetch_data for each ticker.
            
            for ticker in STOCK_TICKERS:
                state = stock_states[ticker] # Get current state for the ticker
                model_info = loaded_models.get(ticker)
                
                stock_update_time = datetime.now().strftime('%H:%M:%S')

                if not model_info:
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: No AI model loaded. Skipping AI signal generation.")
                    continue

                # 1. Fetch live data (OHLCV, Fundamental, News Sentiment)
                df_live, fundamental_data, news_sentiment = fetch_data(ticker, LIVE_DATA_PERIOD, LIVE_DATA_INTERVAL, ALPHAVANTAGE_API_KEY)

                if df_live is None or df_live.empty:
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: Could not fetch recent data or data is insufficient (e.g., outside market hours/API issues).")
                    continue
                
                # 2. Calculate indicators
                df_live_with_indicators = calculate_indicators(df_live)

                # 3. Check for sufficient bars AFTER indicator calculation
                if df_live_with_indicators.empty or len(df_live_with_indicators) < MIN_REQUIRED_BARS_FOR_ANALYSIS:
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: Data too short ({len(df_live_with_indicators) if df_live_with_indicators is not None else 0} bars) after indicator calculation for analysis. Min required: {MIN_REQUIRED_BARS_FOR_ANALYSIS}.")
                    continue
                
                # Get the most current price from the latest fetched and processed dataframe
                current_price_from_df = df_live_with_indicators['close'].iloc[-1]
                state['current_price'] = pd.to_numeric(current_price_from_df, errors='coerce') # Ensure numeric

                if pd.isna(state['current_price']):
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: Current price is NaN after conversion. Skipping further logic for this ticker.")
                    continue

                # 4. Detect Chart Patterns on the live data
                detected_patterns = detect_all_chart_patterns(df_live_with_indicators.copy(), MIN_REQUIRED_BARS_FOR_ANALYSIS)
                
                # 5. Generate AI Signal using all combined info
                ai_signal = generate_ai_signal(df_live_with_indicators, model_info, ticker, fundamental_data, news_sentiment, detected_patterns)
                
                # 6. Generate Actionable Trade Signals
                recent_high = df_live_with_indicators['high'].iloc[-min(20, len(df_live_with_indicators)):].max() # Last 20 periods high
                recent_low = df_live_with_indicators['low'].iloc[-min(20, len(df_live_with_indicators)):].min()   # Last 20 periods low
                
                # Convert AI signal string to integer for generate_trade_signals
                signal_int = {'BUY (AI)': 1, 'SELL (AI)': 0, 'Hold (AI)': 2, 'Hold (AI Model Error)': 2, 'Hold (Insufficient Data)': 2, 'Hold (Feature Mismatch)': 2, 'Hold (No Data)': 2}.get(ai_signal, 2)
                trade_details = generate_trade_signals(signal_int, state['current_price'], fundamental_data, news_sentiment, detected_patterns, recent_high, recent_low)
                
                # --- Decision Logic (Signal Fusion and Trade Management) ---
                # Check for significant price movement since last notification
                price_changed_significantly = False
                if state['last_notification_price'] is not None and state['last_notification_price'] != 0:
                    price_change_percent = abs(state['current_price'] - state['last_notification_price']) / state['last_notification_price']
                    if price_change_percent > 0.005: # Notify if price moved by more than 0.5%
                        price_changed_significantly = True

                if state['in_buy_position']:
                    # Update high_since_entry and trailing stop loss
                    if state['high_since_entry'] is None or state['current_price'] > state['high_since_entry']:
                        state['high_since_entry'] = state['current_price']
                        new_trailing_sl = state['high_since_entry'] * (1 - TRAILING_STOP_LOSS_PERCENTAGE)
                        # Only raise SL if new_trailing_sl is higher than current SL (or initial SL)
                        state['stop_loss_price'] = max(state['stop_loss_price'] if state['stop_loss_price'] is not None else 0, new_trailing_sl)
                        if VERBOSE_MODE:
                            logging.info(f"{stock_update_time} ✅ {ticker}: New high since entry. Trailing SL updated to {state['stop_loss_price']:.2f}")

                    # Exit conditions (ordered by priority: Stop Loss, Take Profit, AI Sell Signal)
                    if state['current_price'] <= state['stop_loss_price']:
                        p_l_percent = (state['current_price'] - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] != 0 else 0
                        notification_message = f"🛑 {ticker}: STOP LOSS HIT! Selling at {state['current_price']:.2f} (Entry: {state['entry_price']:.2f}, SL: {state['stop_loss_price']:.2f}). P/L: {p_l_percent:.2f}%."
                        record_trade(ticker, 'EXIT', state['entry_price'], state['current_price'], current_check_time.strftime('%Y-%m-%d %H:%M:%S'), trade_details)
                        state['in_buy_position'] = False
                        state['entry_price'] = None
                        state['stop_loss_price'] = None
                        state['high_since_entry'] = None
                        state['last_ai_signal'] = 'STOP_LOSS_EXIT'
                        state['last_notification_price'] = state['current_price']
                        logging.info(f"{stock_update_time} {notification_message}")
                        continue # Move to next ticker if position closed

                    elif state['current_price'] >= state['entry_price'] * (1 + TAKE_PROFIT_PERCENTAGE):
                        p_l_percent = (state['current_price'] - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] != 0 else 0
                        notification_message = f"🎯 {ticker}: TAKE PROFIT TARGET HIT! Selling at {state['current_price']:.2f} (Entry: {state['entry_price']:.2f}, TP: {state['entry_price'] * (1 + TAKE_PROFIT_PERCENTAGE):.2f}). P/L: {p_l_percent:.2f}%."
                        record_trade(ticker, 'EXIT', state['entry_price'], state['current_price'], current_check_time.strftime('%Y-%m-%d %H:%M:%S'), trade_details)
                        state['in_buy_position'] = False
                        state['entry_price'] = None
                        state['stop_loss_price'] = None
                        state['high_since_entry'] = None
                        state['last_ai_signal'] = 'TAKE_PROFIT_EXIT'
                        state['last_notification_price'] = state['current_price']
                        logging.info(f"{stock_update_time} {notification_message}")
                        continue # Move to next ticker if position closed

                    elif ai_signal == 'SELL (AI)':
                        p_l_percent = (state['current_price'] - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] != 0 else 0
                        profit_loss_status = "Profit" if p_l_percent >= 0 else "Loss"
                        notification_message = f"💰 {ticker}: AI SELL SIGNAL! Exiting BUY position for {profit_loss_status} at {state['current_price']:.2f} (Entry: {state['entry_price']:.2f}, SL: {state['stop_loss_price']:.2f}). P/L: {p_l_percent:.2f}%."
                        record_trade(ticker, 'EXIT', state['entry_price'], state['current_price'], current_check_time.strftime('%Y-%m-%d %H:%M:%S'), trade_details)
                        state['in_buy_position'] = False
                        state['entry_price'] = None
                        state['stop_loss_price'] = None
                        state['high_since_entry'] = None
                        state['last_ai_signal'] = 'SELL (AI) - EXIT'
                        state['last_notification_price'] = state['current_price']
                        logging.info(f"{stock_update_time} {notification_message}")
                        continue # Move to next ticker if position closed
                    
                    else: # Still in position, no exit signal
                        # Only notify if AI signal changed or price changed significantly
                        if ai_signal != state['last_ai_signal'] or price_changed_significantly:
                            p_l_percent = (state['current_price'] - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] != 0 else 0
                            notification_message = f"🟢 {ticker}: AI says {ai_signal}. Still in BUY position since {state['entry_price']:.2f}. Current: {state['current_price']:.2f} (SL: {state['stop_loss_price']:.2f}). Current P/L: {p_l_percent:.2f}%."
                            state['last_ai_signal'] = ai_signal # Update AI signal state
                            state['last_notification_price'] = state['current_price'] # Update notification price
                            logging.info(f"{stock_update_time} {notification_message}")

                # --- Entry Logic (if not in position) ---
                else: # Not in a buy position
                    
                    # Entry logic is driven by AI signal, and reinforced by actionable trade details
                    if ai_signal == 'BUY (AI)':
                        if not state['in_buy_position']:
                            entry_price = state['current_price']
                            if entry_price is None or not isinstance(entry_price, (int, float)):
                                logging.error(f"  [Error {ticker}] Invalid current_price for AI BUY signal: {entry_price}. Skipping trade entry.")
                                continue

                            stop_loss = trade_details['stop_loss']
                            target_msg = f"TP1: {trade_details['take_profit_1']:.2f}, TP2: {trade_details['take_profit_2']:.2f}."
                            notification_message = f"✅ {ticker}: AI BUY SIGNAL! Consider entering at {entry_price:.2f}. Set Initial Stop Loss at {stop_loss:.2f}. {target_msg} Rationale: {'; '.join(trade_details['rationale'])}"
                            
                            record_trade(ticker, 'ENTER', entry_price, signal_time=current_check_time.strftime('%Y-%m-%d %H:%M:%S'), trade_details=trade_details)
                            state['in_buy_position'] = True
                            state['entry_price'] = entry_price
                            state['stop_loss_price'] = stop_loss
                            state['high_since_entry'] = entry_price # Initialize high_since_entry
                            state['last_ai_signal'] = 'BUY (AI)'
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

                    elif ai_signal == 'SELL (AI)':
                        # If not in position and AI says SELL, it's a bearish signal or stay out
                        if ai_signal != state['last_ai_signal'] or price_changed_significantly:
                            notification_message = f"🔴 {ticker}: AI SELL SIGNAL. Not in position. Staying out for now. Current price: {state['current_price']:.2f}. Rationale: {'; '.join(trade_details['rationale'])}"
                            state['last_ai_signal'] = ai_signal
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

                    elif ai_signal == 'Hold (AI)':
                        # If not in position and AI says HOLD, and no other patterns are active
                        if ai_signal != state['last_ai_signal'] or price_changed_significantly:
                            notification_message = f"🔵 {ticker}: AI says HOLD. Current price: {state['current_price']:.2f}. Not in position. Rationale: {'; '.join(trade_details['rationale'])}"
                            state['last_ai_signal'] = ai_signal
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

            # Print current trade log after all tickers are processed
            print_trade_log()
            
            # Wait for the next interval
            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logging.info("\n\n--- Bot Interrupted by User (Ctrl+C) ---")
        print_trade_log()
        logging.info("Exiting monitoring mode.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n\n--- An unexpected error occurred: {e} ---")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print_trade_log()
        logging.info("Exiting monitoring mode due to error.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|monitor]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'train':
        logging.info("\n--- Starting Model Training Mode ---")
        for ticker in STOCK_TICKERS:
            logging.info(f"\n===== Processing {ticker} for Training =====")
            # Fetch historical data for training. Fundamental and news data will be mostly empty historically.
            df_train, _, _ = fetch_data(ticker, TRAIN_DATA_PERIOD, TRAIN_DATA_INTERVAL, ALPHAVANTAGE_API_KEY)

            if df_train is not None and not df_train.empty:
                # Ensure enough data for indicators *before* calculating them
                if len(df_train) < MIN_REQUIRED_BARS_FOR_ANALYSIS:
                    logging.warning(f"  [Train] Not enough raw data for {ticker} ({len(df_train)} bars). Min required: {MIN_REQUIRED_BARS_FOR_ANALYSIS}. Skipping training.")
                    continue

                df_train_with_indicators = calculate_indicators(df_train)

                if df_train_with_indicators is None or df_train_with_indicators.empty:
                    logging.warning(f"  [Train] Failed to calculate indicators or DataFrame is empty for {ticker} after indicator calculation. Model training aborted for this ticker.")
                else:
                    # After indicator calculation, ensure enough data remains for target creation and model training
                    if len(df_train_with_indicators) < MIN_REQUIRED_BARS_FOR_ANALYSIS:
                        logging.warning(f"  [Train] Not enough data for {ticker} after indicator calculation ({len(df_train_with_indicators)} bars). Min required: {MIN_REQUIRED_BARS_FOR_ANALYSIS}. Skipping training.")
                        continue

                    df_final = create_target_variable(df_train_with_indicators, FUTURE_PERIODS, UP_THRESHOLD, DOWN_THRESHOLD)

                    if df_final is not None and not df_final.empty and 'target' in df_final.columns and df_final['target'].nunique() >= 2:
                        train_and_save_model(df_final, ticker)
                    else:
                        logging.warning(f"  [Train] Not enough labeled data (or not enough distinct classes) to train the model for {ticker} after all preprocessing steps.")
                        logging.warning("  Consider checking data fetching, indicator calculation, or target creation thresholds.")
            else:
                logging.warning(f"  [Train] Training data could not be fetched or was empty for {ticker}. Model training aborted for this ticker.")
        logging.info("\n--- Model Training Complete ---")

    elif mode == 'monitor':
        logging.info("\n--- Starting Live Market Monitoring Mode ---")
        loaded_models = load_models_and_features()
        monitor_markets(loaded_models)

    else:
        print(f"Invalid mode: {mode}. Please use 'train' or 'monitor'.")
        sys.exit(1)
