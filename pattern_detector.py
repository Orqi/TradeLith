import pandas as pd
import numpy as np
import scipy.signal
from sklearn.linear_model import LinearRegression
import logging
from config import VERBOSE_MODE

def find_peaks_and_troughs(prices: pd.Series, order: int = 5) -> tuple[np.ndarray, np.ndarray]:

    prices = pd.to_numeric(prices, errors='coerce').dropna()
    if prices.empty:
        return np.array([]), np.array([])

    peaks, _ = scipy.signal.find_peaks(prices, distance=order, prominence=prices.std() * 0.05)
    troughs, _ = scipy.signal.find_peaks(-prices, distance=order, prominence=prices.std() * 0.05)
    
    return peaks, troughs

def fit_trendline(prices: pd.Series, indices: np.ndarray) -> tuple[float, float, float]:
  
    if len(indices) < 2:
        return 0, 0, 0

    valid_indices = [idx for idx in indices if 0 <= idx < len(prices)]
    if len(valid_indices) < 2:
        return 0, 0, 0

    x = np.array(valid_indices).reshape(-1, 1)
    y = prices.iloc[valid_indices].values

    if len(np.unique(y)) == 1:
        return 0, y[0], 1.0

    model = LinearRegression()
    try:
        model.fit(x, y)
    except ValueError as e:
        if VERBOSE_MODE:
            logging.warning(f"  [Trendline Fit Error] Could not fit trendline: {e}. X shape: {x.shape}, Y shape: {y.shape}.")
        return 0, 0, 0
        
    y_pred = model.predict(x)
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_residual = ((y - y_pred) ** 2).sum()
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    return model.coef_[0], model.intercept_, r_squared

def is_flat(line_slope, tolerance=0.0001):
 
    return abs(line_slope) < tolerance

def is_converging(slope1, slope2, is_rising_pattern=True):

    if is_rising_pattern:
        return slope1 > 0 and slope2 > 0 and slope1 > slope2
    else:
        return slope1 < 0 and slope2 < 0 and abs(slope2) > abs(slope1)

def detect_double_top_bottom(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, price_tolerance: float = 0.01) -> dict:
    patterns = {'has_double_top': False, 'has_double_bottom': False}
    if len(df) < 50: return patterns
    close_prices = df['close']
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            p1_idx_arr = peaks[i]
            p2_idx_arr = peaks[i+1]
            if p1_idx_arr >= len(close_prices) or p2_idx_arr >= len(close_prices): continue
            p1_price = close_prices.iloc[p1_idx_arr]
            p2_price = close_prices.iloc[p2_idx_arr]
            if p2_idx_arr - p1_idx_arr > 5:
                troughs_in_between = [t for t in troughs if p1_idx_arr < t < p2_idx_arr]
                if troughs_in_between:
                    min_price_in_between = close_prices.iloc[troughs_in_between].min()
                    if (abs(p1_price - p2_price) / p1_price < price_tolerance and min_price_in_between < p1_price * 0.95):
                        patterns['has_double_top'] = True
                        if VERBOSE_MODE: logging.info(f"  Double Top detected!")
                        break
    if len(troughs) >= 2:
        for i in range(len(troughs) - 1):
            t1_idx_arr = troughs[i]
            t2_idx_arr = troughs[i+1]
            if t1_idx_arr >= len(close_prices) or t2_idx_arr >= len(close_prices): continue
            t1_price = close_prices.iloc[t1_idx_arr]
            t2_price = close_prices.iloc[t2_idx_arr]
            if t2_idx_arr - t1_idx_arr > 5:
                peaks_in_between = [p for p in peaks if t1_idx_arr < p < t2_idx_arr]
                if peaks_in_between:
                    max_price_in_between = close_prices.iloc[peaks_in_between].max()
                    if (abs(t1_price - t2_price) / t1_price < price_tolerance and max_price_in_between > t1_price * 1.05):
                        patterns['has_double_bottom'] = True
                        if VERBOSE_MODE: logging.info(f"  Double Bottom detected!")
                        break
    return patterns

def detect_triple_top_bottom(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, price_tolerance: float = 0.015) -> dict:
    patterns = {'has_triple_top': False, 'has_triple_bottom': False}
    if len(df) < 75: return patterns
    close_prices = df['close']
    if len(peaks) >= 3:
        for i in range(len(peaks) - 2):
            p1_idx_arr, p2_idx_arr, p3_idx_arr = peaks[i], peaks[i+1], peaks[i+2]
            if p3_idx_arr >= len(close_prices): continue
            p1_price, p2_price, p3_price = close_prices.iloc[p1_idx_arr], close_prices.iloc[p2_idx_arr], close_prices.iloc[p3_idx_arr]
            if (p2_idx_arr - p1_idx_arr > 5 and p3_idx_arr - p2_idx_arr > 5 and
                abs(p1_price - p2_price) / p1_price < price_tolerance and
                abs(p2_price - p3_price) / p3_price < price_tolerance):
                troughs_in_between = [t for t in troughs if t > p1_idx_arr and t < p3_idx_arr]
                if len(troughs_in_between) >= 2:
                    patterns['has_triple_top'] = True
                    if VERBOSE_MODE: logging.info(f"  Triple Top detected!")
                    break
    if len(troughs) >= 3:
        for i in range(len(troughs) - 2):
            t1_idx_arr, t2_idx_arr, t3_idx_arr = troughs[i], troughs[i+1], troughs[i+2]
            if t3_idx_arr >= len(close_prices): continue
            t1_price, t2_price, t3_price = close_prices.iloc[t1_idx_arr], close_prices.iloc[t2_idx_arr], close_prices.iloc[t3_idx_arr]
            if (t2_idx_arr - t1_idx_arr > 5 and t3_idx_arr - t2_idx_arr > 5 and
                abs(t1_price - t2_price) / t1_price < price_tolerance and
                abs(t2_price - t3_price) / t3_price < price_tolerance):
                peaks_in_between = [p for p in peaks if p > t1_idx_arr and p < t3_idx_arr]
                if len(peaks_in_between) >= 2:
                    patterns['has_triple_bottom'] = True
                    if VERBOSE_MODE: logging.info(f"  Triple Bottom detected!")
                    break
    return patterns

def detect_head_and_shoulders_patterns(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> dict:
    patterns = {'has_head_and_shoulders': False, 'has_inverse_head_and_shoulders': False}
    if len(df) < 100: return patterns
    close_prices = df['close']
    if len(peaks) >= 3 and len(troughs) >= 2:
        for i in range(len(peaks) - 2):
            ls_idx, h_idx, rs_idx = peaks[i], peaks[i+1], peaks[i+2]
            if rs_idx >= len(close_prices): continue
            ls_price, h_price, rs_price = close_prices.iloc[ls_idx], close_prices.iloc[h_idx], close_prices.iloc[rs_idx]
            if (h_price > ls_price and h_price > rs_price and
                abs(ls_price - rs_price) / ls_price < 0.03 and
                h_price > ls_price * 1.05 and h_price > rs_price * 1.05 and
                ls_idx < h_idx < rs_idx):
                trough1_candidates = [t for t in troughs if ls_idx < t < h_idx]
                trough2_candidates = [t for t in troughs if h_idx < t < rs_idx]
                if trough1_candidates and trough2_candidates:
                    neckline_trough1_idx = trough1_candidates[-1]
                    neckline_trough2_idx = trough2_candidates[0]
                    slope, _, r_squared = fit_trendline(close_prices, np.array([neckline_trough1_idx, neckline_trough2_idx]))
                    if is_flat(slope, tolerance=0.005) and r_squared > 0.5:
                        patterns['has_head_and_shoulders'] = True
                        if VERBOSE_MODE: logging.info(f"  Head and Shoulders detected!")
                        break
    if len(troughs) >= 3 and len(peaks) >= 2:
        for i in range(len(troughs) - 2):
            ls_idx, h_idx, rs_idx = troughs[i], troughs[i+1], troughs[i+2]
            if rs_idx >= len(close_prices): continue
            ls_price, h_price, rs_price = close_prices.iloc[ls_idx], close_prices.iloc[h_idx], close_prices.iloc[rs_idx]
            if (h_price < ls_price and h_price < rs_price and
                abs(ls_price - rs_price) / ls_price < 0.03 and
                h_price < ls_price * 0.95 and h_price < rs_price * 0.95 and
                ls_idx < h_idx < rs_idx):
                peak1_candidates = [p for p in peaks if ls_idx < p < h_idx]
                peak2_candidates = [p for p in peaks if h_idx < p < rs_idx]
                if peak1_candidates and peak2_candidates:
                    neckline_peak1_idx = peak1_candidates[-1]
                    neckline_peak2_idx = peak2_candidates[0]
                    slope, _, r_squared = fit_trendline(close_prices, np.array([neckline_peak1_idx, neckline_peak2_idx]))
                    if is_flat(slope, tolerance=0.005) and r_squared > 0.5:
                        patterns['has_inverse_head_and_shoulders'] = True
                        if VERBOSE_MODE: logging.info(f"  Inverse Head and Shoulders detected!")
                        break
    return patterns

def detect_wedge_patterns(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, lookback_window: int = 50) -> dict:
    patterns = {'has_rising_wedge': False, 'has_falling_wedge': False}
    if len(df) < lookback_window: return patterns
    close_prices = df['close']
    recent_peaks = sorted([p for p in peaks if p >= len(df) - lookback_window])
    recent_troughs = sorted([t for t in troughs if t >= len(df) - lookback_window])
    if len(recent_peaks) < 2 or len(recent_troughs) < 2: return patterns
    upper_slope, _, upper_r_sq = fit_trendline(close_prices, np.array(recent_peaks))
    lower_slope, _, lower_r_sq = fit_trendline(close_prices, np.array(recent_troughs))
    if upper_r_sq > 0.6 and lower_r_sq > 0.6:
        if upper_slope > 0 and lower_slope > 0 and upper_slope > lower_slope * 1.2 and is_converging(upper_slope, lower_slope, True):
            patterns['has_rising_wedge'] = True
            if VERBOSE_MODE: logging.info(f"  Rising Wedge detected!")
    if upper_r_sq > 0.6 and lower_r_sq > 0.6:
        if upper_slope < 0 and lower_slope < 0 and abs(lower_slope) > abs(upper_slope) * 1.2 and is_converging(upper_slope, lower_slope, False):
            patterns['has_falling_wedge'] = True
            if VERBOSE_MODE: logging.info(f"  Falling Wedge detected!")
    return patterns

def detect_triangle_patterns(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, lookback_window: int = 50) -> dict:
    patterns = {'has_ascending_triangle': False, 'has_descending_triangle': False, 'has_symmetrical_triangle': False}
    if len(df) < lookback_window: return patterns
    close_prices = df['close']
    recent_peaks = sorted([p for p in peaks if p >= len(df) - lookback_window])
    recent_troughs = sorted([t for t in troughs if t >= len(df) - lookback_window])
    if len(recent_peaks) < 2 or len(recent_troughs) < 2: return patterns
    upper_slope, _, upper_r_sq = fit_trendline(close_prices, np.array(recent_peaks))
    lower_slope, _, lower_r_sq = fit_trendline(close_prices, np.array(recent_troughs))
    if upper_r_sq > 0.6 and lower_r_sq > 0.6:
        if is_flat(upper_slope, tolerance=0.0005) and lower_slope > 0.0005:
            patterns['has_ascending_triangle'] = True
            if VERBOSE_MODE: logging.info(f"  Ascending Triangle detected!")
        if is_flat(lower_slope, tolerance=0.0005) and upper_slope < -0.0005:
            patterns['has_descending_triangle'] = True
            if VERBOSE_MODE: logging.info(f"  Descending Triangle detected!")
        if upper_slope < -0.0005 and lower_slope > 0.0005:
            patterns['has_symmetrical_triangle'] = True
            if VERBOSE_MODE: logging.info(f"  Symmetrical Triangle detected!")
    return patterns

def detect_flag_patterns(df: pd.DataFrame, min_pole_len: int = 10, min_flag_len: int = 5, pole_change_threshold: float = 0.05, consolidation_pct: float = 0.03) -> dict:

    patterns = {'has_bullish_flag': False, 'has_bearish_flag': False}
    if len(df) < (min_pole_len + min_flag_len + 1): return patterns
    close_prices = df['close']
    max_i_for_pole_start = len(close_prices) - min_pole_len - min_flag_len
    if max_i_for_pole_start < 0: return patterns
    for i in range(max_i_for_pole_start + 1):
        pole_start_price = close_prices.iloc[i]
        pole_end_price = close_prices.iloc[i + min_pole_len]
        if (pole_end_price - pole_start_price) / pole_start_price > pole_change_threshold:
            flag_slice = close_prices.iloc[i + min_pole_len : ]
            if len(flag_slice) >= min_flag_len:
                flag_max = flag_slice.max()
                flag_min = flag_slice.min()
                if (flag_max - flag_min) / pole_end_price < consolidation_pct:
                    flag_slope, _, _ = fit_trendline(flag_slice, np.arange(len(flag_slice)))
                    if flag_slope < 0:
                        patterns['has_bullish_flag'] = True
                        if VERBOSE_MODE: logging.info(f"  Bullish Flag detected!")
                        break
        if (pole_end_price - pole_start_price) / pole_start_price < -pole_change_threshold:
            flag_slice = close_prices.iloc[i + min_pole_len : ]
            if len(flag_slice) >= min_flag_len:
                flag_max = flag_slice.max()
                flag_min = flag_slice.min()
                if (flag_max - flag_min) / pole_end_price < consolidation_pct:
                    flag_slope, _, _ = fit_trendline(flag_slice, np.arange(len(flag_slice)))
                    if flag_slope > 0:
                        patterns['has_bearish_flag'] = True
                        if VERBOSE_MODE: logging.info(f"  Bearish Flag detected!")
                        break
    return patterns

def detect_cup_and_handle_pattern(df: pd.DataFrame, lookback_window: int = 60, min_cup_depth_pct: float = 0.10) -> dict:

    patterns = {'has_cup_and_handle': False}
    if len(df) < lookback_window: return patterns
    close_prices = df['close']
    recent_df = df.iloc[-lookback_window:].copy()
    recent_close = recent_df['close']
    cup_trough_idx = recent_close.iloc[:int(len(recent_close)*0.7)].idxmin()
    cup_trough_price = recent_close.loc[cup_trough_idx]
    left_lip_price = recent_close.iloc[0]
    potential_right_lip_slice = recent_close.loc[cup_trough_idx:]
    if potential_right_lip_slice.empty: return patterns
    right_lip_idx = potential_right_lip_slice.idxmax()
    right_lip_price = recent_close.loc[right_lip_idx]
    if (abs(left_lip_price - right_lip_price) / left_lip_price < 0.05 and (left_lip_price - cup_trough_price) / left_lip_price > min_cup_depth_pct):
        handle_start_idx = recent_df.index.get_loc(right_lip_idx)
        if handle_start_idx + 5 >= len(recent_df): return patterns
        handle_slice = recent_df.iloc[handle_start_idx:].copy()
        handle_high = handle_slice['high'].max()
        handle_low = handle_slice['low'].min()
        if handle_slice['close'].iloc[-1] < handle_slice['close'].iloc[0] and (handle_high - handle_low) / right_lip_price < 0.05:
            patterns['has_cup_and_handle'] = True
            if VERBOSE_MODE: logging.info(f"  Cup and Handle detected!")
    return patterns

def detect_all_chart_patterns(df: pd.DataFrame, min_required_bars: int) -> dict:

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

    pattern_detection_window = min(len(df), 100)
    recent_df_for_patterns = df.iloc[-pattern_detection_window:].copy()

    peaks, troughs = find_peaks_and_troughs(recent_df_for_patterns['close'], order=5)

    if recent_df_for_patterns.empty or len(peaks) < 2 or len(troughs) < 2:
        return pattern_flags

    pattern_flags.update(detect_double_top_bottom(recent_df_for_patterns, peaks, troughs))
    pattern_flags.update(detect_triple_top_bottom(recent_df_for_patterns, peaks, troughs))
    pattern_flags.update(detect_head_and_shoulders_patterns(recent_df_for_patterns, peaks, troughs))
    pattern_flags.update(detect_wedge_patterns(recent_df_for_patterns, peaks, troughs))
    pattern_flags.update(detect_triangle_patterns(recent_df_for_patterns, peaks, troughs))
    pattern_flags.update(detect_flag_patterns(recent_df_for_patterns))
    pattern_flags.update(detect_cup_and_handle_pattern(recent_df_for_patterns))
    
    if VERBOSE_MODE:
        detected_true_patterns = {k: v for k, v in pattern_flags.items() if v}
        if detected_true_patterns:
            logging.info(f"[Pattern Detection] Detected patterns for latest data: {detected_true_patterns}")
        else:
            logging.info("[Pattern Detection] No significant patterns detected.")

    return pattern_flags

