import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime

# --- AI/ML Libraries (will be used in later steps) ---
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib # For saving/loading models
# ---------------------------------------------------

# Define the stock tickers you want to analyze
STOCK_TICKERS = ["GRSE.NS", "ZENTEC.BO", "^NSEI", "KITEX.BO", "YESBANK.NS"]

# For "monitoring markets constantly", we need to fetch recent data.
DATA_PERIOD = "1d" # Fetch data for the current day
DATA_INTERVAL = "1m" # Fetch data at 1-minute intervals

# How often the bot should check the market (in seconds)
CHECK_INTERVAL_SECONDS = 300 # Check every 5 minutes

# Dictionary to store the last known signal and price for each stock to avoid re-notifying
last_signals = {ticker: {'signal': 'Hold', 'price': None} for ticker in STOCK_TICKERS}

# --- Placeholder for trained AI model (will be loaded here) ---
# ai_model = None
# AI_MODEL_PATH = "my_trading_model.joblib"
# -----------------------------------------------------------

def fetch_recent_data(ticker, period, interval):
    """
    Fetches recent stock data using yfinance for continuous monitoring.
    """
    try:
        stock_data = yf.download(ticker, period=period, interval=interval, progress=False)
        if stock_data.empty:
            return None

        # Flatten column names for easier use with pandas_ta
        new_columns = []
        for col in stock_data.columns:
            if isinstance(col, tuple):
                new_columns.append(col[0].replace(' ', '_').lower())
            else:
                new_columns.append(col.replace(' ', '_').lower())
        stock_data.columns = new_columns

        return stock_data
    except Exception as e:
        print(f"Error fetching recent data for {ticker}: {e}")
        return None

def calculate_indicators(df):
    """
    Calculates a comprehensive set of technical indicators using pandas_ta.
    """
    if df is None or df.empty:
        return df

    # Basic OHLCV columns must be present and correctly named
    # For pandas_ta, ensure 'open', 'high', 'low', 'close', 'volume' are present
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing one or more required OHLCV columns for indicator calculation. Has: {df.columns.tolist()}, Needs: {required_cols}")
        return df # Return as is, indicators might fail later

    # --- Trend Indicators ---
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=10, append=True) # Exponential Moving Average
    df.ta.ema(length=20, append=True) # Exponential Moving Average
    df.ta.adx(append=True)           # Average Directional Index (measures trend strength)
    df.ta.macd(fast=12, slow=26, signal=9, append=True) # MACD
    df.ta.ichimoku(append=True)      # Ichimoku Cloud (generates many columns)

    # --- Momentum Indicators ---
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)         # Stochastic Oscillator
    df.ta.cci(append=True)           # Commodity Channel Index
    df.ta.mfi(append=True)           # Money Flow Index (includes volume)

    # --- Volatility Indicators ---
    df.ta.bbands(length=20, std=2, append=True) # Bollinger Bands
    df.ta.atr(append=True)           # Average True Range
    df.ta.donchian(append=True)      # Donchian Channels

    # --- Volume Indicators ---
    df.ta.obv(append=True)
    df.ta.vwap(append=True)          # Volume Weighted Average Price (requires valid 'open', 'high', 'low', 'close', 'volume')

    # --- Other/Reversal Indicators ---
    df.ta.psar(append=True)          # Parabolic SAR

    return df

def generate_signals_rules_based(df):
    """
    Generates simple buy/sell/hold signals based on a *combination* of rules.
    This is still rules-based but now tries to combine.
    """
    if df is None or df.empty or len(df) < 30: # Ensure enough data for most indicators
        return 'Hold'

    latest_data = df.iloc[-1]
    prev_data = df.iloc[-2] # For checking crossovers

    # Initialize signal as 'Hold'
    signal = 'Hold'

    # --- Combine SMA and MACD for a stronger trend signal ---
    # Buy condition: SMA 10 crosses above SMA 20 AND MACD line crosses above Signal line
    # Sell condition: SMA 10 crosses below SMA 20 AND MACD line crosses below Signal line
    if 'sma_10' in df.columns and 'sma_20' in df.columns and \
       'macd_12_26_9' in df.columns and 'macds_12_26_9' in df.columns:
        if prev_data['sma_10'] < prev_data['sma_20'] and \
           latest_data['sma_10'] > latest_data['sma_20'] and \
           prev_data['macd_12_26_9'] < prev_data['macds_12_26_9'] and \
           latest_data['macd_12_26_9'] > latest_data['macds_12_26_9']:
            signal = 'STRONG BUY (SMA & MACD)'
        elif prev_data['sma_10'] > prev_data['sma_20'] and \
             latest_data['sma_10'] < latest_data['sma_20'] and \
             prev_data['macd_12_26_9'] > prev_data['macds_12_26_9'] and \
             latest_data['macd_12_26_9'] < latest_data['macds_12_26_9']:
            signal = 'STRONG SELL (SMA & MACD)'

    # --- Add RSI/Stochastic confirmation (momentum) ---
    # If already a BUY signal, confirm with oversold RSI/Stochastic turning up
    # If already a SELL signal, confirm with overbought RSI/Stochastic turning down
    if signal.startswith('BUY') and 'rsi_14' in df.columns and 'stochk_14_3_3' in df.columns:
        if latest_data['rsi_14'] < 50 and latest_data['stochk_14_3_3'] < 50: # RSI/Stoch not overbought
            # If momentum indicators are also supportive of an uptrend after a buy signal
            # This is a basic example; more complex logic needed for real confirmation
            signal += ' + Momentum'
    elif signal.startswith('SELL') and 'rsi_14' in df.columns and 'stochk_14_3_3' in df.columns:
        if latest_data['rsi_14'] > 50 and latest_data['stochk_14_3_3'] > 50: # RSI/Stoch not oversold
            # If momentum indicators are also supportive of a downtrend after a sell signal
            signal += ' + Momentum'

    # --- Fallback/Override based on strong RSI/Stochastic signals (independent) ---
    # If no strong trend signal, but clear overbought/oversold condition
    if signal == 'Hold':
        if 'rsi_14' in df.columns:
            if latest_data['rsi_14'] > 70: # More strict overbought
                signal = 'SELL (RSI Overbought)'
            elif latest_data['rsi_14'] < 30: # More strict oversold
                signal = 'BUY (RSI Oversold)'
        # If no RSI signal, check Stochastic
        if signal == 'Hold' and 'stochk_14_3_3' in df.columns:
            if latest_data['stochk_14_3_3'] > 80:
                signal = 'SELL (Stoch Overbought)'
            elif latest_data['stochk_14_3_3'] < 20:
                signal = 'BUY (Stoch Oversold)'

    # --- Consider Bollinger Bands for volatility and potential reversals ---
    # If price moves outside bands, it might signal a reversal or strong continuation
    # This is more complex; usually combined with other indicators
    # For simplicity, let's just add a note if price is near band
    if 'bbu_5_2.0' in df.columns and 'bbl_5_2.0' in df.columns:
        if latest_data['close'] > latest_data['bbu_5_2.0']:
            if signal.startswith('BUY'): signal = 'BUY (Bands Broken UP, but cautious)'
            elif signal.startswith('Hold'): signal = 'CONSIDER SELL (Price Above BBANDS)'
        elif latest_data['close'] < latest_data['bbl_5_2.0']:
            if signal.startswith('SELL'): signal = 'SELL (Bands Broken DOWN, but cautious)'
            elif signal.startswith('Hold'): signal = 'CONSIDER BUY (Price Below BBANDS)'


    return signal

# --- Placeholder for AI-driven signal generation (Future Step) ---
# def generate_signals_ai_driven(df, model):
#     """
#     Generates buy/sell/hold signals using a trained AI model.
#     Requires a pre-trained model and proper feature engineering.
#     """
#     if df is None or df.empty or model is None:
#         return 'Hold (AI Model Not Ready)'

#     # 1. Prepare features for the AI model from the latest data
#     # This is highly dependent on how your AI model was trained.
#     # For example, you might need a DataFrame with only the indicator columns
#     # from the last row, reshaped correctly.
#     # `feature_vector = df[FEATURES_FOR_AI].iloc[-1].values.reshape(1, -1)`
#     # Make sure to handle NaNs if your model can't.

#     # 2. Make a prediction
#     # prediction = model.predict(feature_vector)[0]
#     # prediction_proba = model.predict_proba(feature_vector)[0] # For confidence

#     # 3. Map prediction to signal
#     # if prediction == 1: return 'BUY (AI)'
#     # elif prediction == 0: return 'SELL (AI)' # Or 0 could be Hold, depending on training
#     # else: return 'Hold (AI)' # Example mapping
#     return 'Hold (AI Integration Pending)' # Placeholder

# --- End AI-driven signal generation placeholder ---


def monitor_markets():
    """
    Continuously monitors the specified stocks and provides buy/sell notifications.
    """
    print(f"\nStarting real-time market monitor for: {STOCK_TICKERS}")
    print(f"Checking every {CHECK_INTERVAL_SECONDS} seconds...")

    # Load AI model if available (future step)
    # global ai_model
    # try:
    #     ai_model = joblib.load(AI_MODEL_PATH)
    #     print(f"AI model loaded from {AI_MODEL_PATH}")
    # except FileNotFoundError:
    #     print("No AI model found. Running with rules-based signals.")
    #     ai_model = None
    # except Exception as e:
    #     print(f"Error loading AI model: {e}. Running with rules-based signals.")
    #     ai_model = None

    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n--- Checking market at {timestamp} ---")

        for ticker in STOCK_TICKERS:
            df = fetch_recent_data(ticker, DATA_PERIOD, DATA_INTERVAL)

            if df is not None and not df.empty:
                df = calculate_indicators(df)
                
                # Use the enhanced rules-based signal for now
                current_signal = generate_signals_rules_based(df)
                
                # --- Future AI integration point ---
                # if ai_model:
                #     ai_signal = generate_signals_ai_driven(df, ai_model)
                #     # Decide how to combine rules-based and AI signals
                #     # For now, let's prioritize AI if it gives a strong signal
                #     if ai_signal != 'Hold (AI Integration Pending)':
                #         current_signal = f"{current_signal} / AI: {ai_signal}"
                # -----------------------------------

                current_price = df['close'].iloc[-1] if 'close' in df.columns and not df['close'].empty else 'N/A'

                # Check if signal has changed or price has moved significantly
                # (Only if price is a valid number)
                price_changed_significantly = False
                if last_signals[ticker]['price'] is not None and isinstance(current_price, (int, float)):
                    if current_price != 0: # Avoid division by zero
                        price_changed_significantly = abs(current_price - last_signals[ticker]['price']) / current_price > 0.005 # 0.5% change

                if current_signal != last_signals[ticker]['signal'] or price_changed_significantly:
                    print(f"🚨 {ticker}: NEW STATUS! {current_signal} (Close: {current_price:.2f})")
                    last_signals[ticker]['signal'] = current_signal
                    last_signals[ticker]['price'] = current_price
                else:
                    print(f"🟢 {ticker}: {current_signal} (Close: {current_price:.2f})")
            else:
                print(f"⚠️ {ticker}: Could not fetch recent data or data is insufficient for analysis (e.g., outside market hours).")

        print(f"\n--- Next check in {CHECK_INTERVAL_SECONDS} seconds ---")
        time.sleep(CHECK_INTERVAL_SECONDS)

def get_current_recommendation(ticker_to_check=None):
    """
    Provides the latest recommendation for a specific ticker or all monitored tickers.
    This function simulates a user asking the bot.
    """
    print("\n--- Current Recommendations (On Demand) ---")
    tickers_to_report = []
    if ticker_to_check and ticker_to_check.upper() in [t.upper() for t in STOCK_TICKERS]:
        tickers_to_report = [ticker_to_check.upper()]
    else:
        tickers_to_report = STOCK_TICKERS # Report on all if no specific one is asked

    for ticker in tickers_to_report:
        df = fetch_recent_data(ticker, DATA_PERIOD, DATA_INTERVAL)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            signal = generate_signals_rules_based(df) # Using rules-based for now
            current_price = df['close'].iloc[-1] if 'close' in df.columns and not df['close'].empty else 'N/A'
            print(f"{ticker}: {signal} (Current Price: {current_price:.2f})")
        else:
            print(f"{ticker}: Data not available or insufficient for a recommendation.")

if __name__ == "__main__":
    monitor_markets()

    # You can stop the above loop (Ctrl+C) and then manually call:
    # get_current_recommendation("GRSE.NS")
    # get_current_recommendation()