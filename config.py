# config.py
import os
import logging

# Configure logging for better output management
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Market Configuration ---
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
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_ALPHA_VANTAGE_API_KEY") # Use environment variable if set

# --- AI Feature Configuration ---
# Features to always include in the model, even if they are constant or often NaN
ALWAYS_INCLUDED_FEATURES = [
    'pe_ratio', 'market_capitalization', 'book_value_per_share', 'dividend_yield', # Fundamental Data
    'avg_news_sentiment', # News Sentiment
    'has_double_top', 'has_double_bottom', 'has_triple_top', 'has_triple_bottom',
    'has_head_and_shoulders', 'has_inverse_head_and_shoulders',
    'has_rising_wedge', 'has_falling_wedge',
    'has_ascending_triangle', 'has_descending_triangle', 'has_symmetrical_triangle',
    'has_bullish_flag', 'has_bearish_flag', 'has_cup_and_handle' # Chart Patterns
]

# --- Output & Logging ---
WEB_UI_OUTPUT_PATH = "web-ui/output.json"
VERBOSE_MODE = True # Set to False for less console output

# --- Global State (for the monitoring process) ---
# NOTE: In a real production system, this would be persisted in a database (e.g., SQLite, PostgreSQL)
# For this example, it's a dictionary in memory, which means state is lost on bot restart.
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
