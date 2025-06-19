# main.py (Updated Entry Point)
import sys
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information.", category=UserWarning)
warnings.filterwarnings("ignore", message="The behavior of DataFrame.idxmin with a float dtype index is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="The behavior of DataFrame.idxmax with a float dtype index is deprecated.*", category=FutureWarning)

from config import (
    STOCK_TICKERS, TRAIN_DATA_PERIOD, TRAIN_DATA_INTERVAL, FUTURE_PERIODS,
    UP_THRESHOLD, DOWN_THRESHOLD, MIN_REQUIRED_BARS_FOR_ANALYSIS
)
from data_handler import fetch_data
from indicator_calculator import calculate_indicators
from feature_processor import create_target_variable
from model_manager import train_and_save_model
from monitor_loop import run_monitor_loop # Import the monitoring loop function


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|monitor]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'train':
        logging.info("\n--- Starting Model Training Mode ---")
        for ticker in STOCK_TICKERS:
            logging.info(f"\n===== Processing {ticker} for Training =====")
            df_train, _, _ = fetch_data(ticker, TRAIN_DATA_PERIOD, TRAIN_DATA_INTERVAL)

            if df_train is not None and not df_train.empty:
                if len(df_train) < MIN_REQUIRED_BARS_FOR_ANALYSIS:
                    logging.warning(f"  [Train] Not enough raw data for {ticker} ({len(df_train)} bars). Min required: {MIN_REQUIRED_BARS_FOR_ANALYSIS}. Skipping training.")
                    continue

                df_train_with_indicators = calculate_indicators(df_train)

                if df_train_with_indicators is None or df_train_with_indicators.empty:
                    logging.warning(f"  [Train] Failed to calculate indicators or DataFrame is empty for {ticker} after indicator calculation. Model training aborted for this ticker.")
                else:
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
        run_monitor_loop() # Call the centralized monitoring loop function

    else:
        print(f"Invalid mode: {mode}. Please use 'train' or 'monitor'.")
        sys.exit(1)
