import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

STOCK_TICKERS = ["GRSE.NS", "ZENTEC.NS", "YESBANK.NS", "KITEX.NS"]

TRAIN_DATA_PERIOD = "10y"
TRAIN_DATA_INTERVAL = "1d"


LIVE_DATA_PERIOD = "60d" 
LIVE_DATA_INTERVAL = "1h"


CHECK_INTERVAL_SECONDS = 60


MIN_REQUIRED_BARS_FOR_ANALYSIS = 200

FUTURE_PERIODS = 1
UP_THRESHOLD = 0.005
DOWN_THRESHOLD = -0.005

MODEL_DIR = "trained_models"
PARAM_GRID = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

INITIAL_STOP_LOSS_PERCENTAGE = 0.02
TRAILING_STOP_LOSS_PERCENTAGE = 0.015
TAKE_PROFIT_PERCENTAGE = 0.03


ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "V48L9DEA6MTTLLIO")

ALWAYS_INCLUDED_FEATURES = [
    'pe_ratio', 'market_capitalization', 'book_value_per_share', 'dividend_yield',
    'avg_news_sentiment',
    'has_double_top', 'has_double_bottom', 'has_triple_top', 'has_triple_bottom',
    'has_head_and_shoulders', 'has_inverse_head_and_shoulders',
    'has_rising_wedge', 'has_falling_wedge',
    'has_ascending_triangle', 'has_descending_triangle', 'has_symmetrical_triangle',
    'has_bullish_flag', 'has_bearish_flag', 'has_cup_and_handle'
]

WEB_UI_OUTPUT_PATH = "web-ui/output.json"
VERBOSE_MODE = True

stock_states = {
    ticker: {
        'last_ai_signal': 'Hold',
        'current_price': None,
        'in_buy_position': False,
        'entry_price': None,
        'stop_loss_price': None,
        'high_since_entry': None,
        'last_notification_price': None
    } for ticker in STOCK_TICKERS
}

trade_log = []
