import pandas as pd
import numpy as np
import logging
from config import VERBOSE_MODE, ALWAYS_INCLUDED_FEATURES

def create_target_variable(df: pd.DataFrame, future_periods: int, up_threshold: float, down_threshold: float) -> pd.DataFrame | None:

    if df is None or df.empty:
        if VERBOSE_MODE:
            logging.warning("[Target Creation] Input DataFrame is empty or None. Cannot create target.")
        return None

    if 'close' not in df.columns or not pd.api.types.is_numeric_dtype(df['close']):
        if VERBOSE_MODE:
            logging.warning("[Target Creation Warning] 'close' column is missing or not numeric. Cannot create target variable.")
        return None

    df_copy = df.copy()
    
    df_copy['future_close'] = df_copy['close'].shift(-future_periods)
    df_copy['price_change'] = (df_copy['future_close'] - df_copy['close']) / df_copy['close']

    df_copy['target'] = 2
    df_copy.loc[df_copy['price_change'] >= up_threshold, 'target'] = 1
    df_copy.loc[df_copy['price_change'] <= down_threshold, 'target'] = 0

    df_copy.drop(columns=['future_close', 'price_change'], inplace=True)
    
    initial_rows = df_copy.shape[0]
    df_copy.dropna(subset=['target'], inplace=True)
    if VERBOSE_MODE and (initial_rows - df_copy.shape[0]) > 0:
        logging.info(f"[Target Creation] Dropped {initial_rows - df_copy.shape[0]} rows due to NaN in target variable (last {future_periods} rows).")

    df_copy['target'] = df_copy['target'].astype(int)

    if VERBOSE_MODE:
        logging.info(f"[Target Creation] Label distribution (0=Sell, 1=Buy, 2=Hold):\n{df_copy['target'].value_counts()}")
        logging.info(f"[Target Creation] Shape after target creation: {df_copy.shape}")
    return df_copy

def prepare_features_for_ai(df: pd.DataFrame, fundamental_data: dict = None, news_sentiment: dict = None, for_training: bool = True, known_features: list = None) -> tuple[pd.DataFrame, pd.Series | None, list | None]:

    logging.info("[Feature Prep] Preparing features for AI model...")

    df_processed = df.copy()

    if fundamental_data:
        for key, value in fundamental_data.items():
            df_processed[key] = value
    if news_sentiment:
        for key, value in news_sentiment.items():
            df_processed[key] = value

    pattern_feature_names = [
        'has_double_top', 'has_double_bottom', 'has_triple_top', 'has_triple_bottom',
        'has_head_and_shoulders', 'has_inverse_head_and_shoulders',
        'has_rising_wedge', 'has_falling_wedge',
        'has_ascending_triangle', 'has_descending_triangle', 'has_symmetrical_triangle',
        'has_bullish_flag', 'has_bearish_flag', 'has_cup_and_handle'
    ]
    for pattern_name in pattern_feature_names:
        if pattern_name not in df_processed.columns:
            df_processed[pattern_name] = 0

    excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'target'] 
    
    X = pd.DataFrame()
    y = None
    final_features_list = None

    if for_training:
        all_potential_features = [col for col in df_processed.columns if df_processed[col].dtype in ['float64', 'int64'] and col not in excluded_cols]

        features_to_use = [f for f in ALWAYS_INCLUDED_FEATURES if f in df_processed.columns]

        for f in all_potential_features:
            if f not in features_to_use:
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

        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    col_mean = X[col].mean()
                    if pd.isna(col_mean):
                        X[col].fillna(0, inplace=True)
                    else:
                        X[col].fillna(col_mean, inplace=True)
                else: 
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
        
        if y is not None:
            combined_df = X.copy()
            combined_df['target'] = y
            combined_df.dropna(subset=['target'], inplace=True)
            X = combined_df[features_to_use]
            y = combined_df['target']
            logging.info(f"  [Feature Prep] Shape after target NaN drop: {X.shape}")

    else:
        if known_features is None:
            logging.error("[Feature Prep] 'known_features' must be provided for prediction mode.")
            return pd.DataFrame(), None, None

        X_pred_dict = {}
        for feature in known_features:
            if feature in df_processed.columns:
                value = df_processed[feature].iloc[-1]
                if pd.api.types.is_bool_dtype(df_processed[feature]):
                    X_pred_dict[feature] = int(value)
                elif pd.isna(value):
                    X_pred_dict[feature] = 0
                else:
                    X_pred_dict[feature] = value
            else:
                X_pred_dict[feature] = 0
        
        X = pd.DataFrame([X_pred_dict])
        X.fillna(0, inplace=True)

        final_features_list = known_features

    logging.info(f"[Feature Prep] Features prepared. Final feature shape: {X.shape}")
    return X, y, final_features_list

