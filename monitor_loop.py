# monitor_loop.py
import time
from datetime import datetime
import logging
import sys
import pandas as pd
from config import (
    CHECK_INTERVAL_SECONDS, STOCK_TICKERS, LIVE_DATA_PERIOD, LIVE_DATA_INTERVAL,
    MIN_REQUIRED_BARS_FOR_ANALYSIS, TAKE_PROFIT_PERCENTAGE, VERBOSE_MODE,
    stock_states, trade_log, INITIAL_STOP_LOSS_PERCENTAGE, TRAILING_STOP_LOSS_PERCENTAGE
)
from data_handler import fetch_data
from indicator_calculator import calculate_indicators
from pattern_detector import detect_all_chart_patterns
from model_manager import generate_ai_signal, load_models_and_features
from trade_manager import generate_trade_signals, record_trade, print_trade_log
from output_writer import save_output_to_json

def run_monitor_loop():
    """
    Continuously monitors markets, fetches data, generates signals, and manages trades.
    """
    logging.info(f"\nStarting real-time market monitor for: {STOCK_TICKERS}")
    logging.info(f"Checking every {CHECK_INTERVAL_SECONDS} seconds...")

    loaded_models = load_models_and_features()

    if not loaded_models:
        logging.warning("WARNING: No trained models found. Cannot provide AI-driven signals. Please run 'train' mode first.")
        logging.info("Exiting monitoring mode as no AI models are available.")
        return

    for ticker in STOCK_TICKERS:
        if ticker not in stock_states:
            stock_states[ticker] = {
                'last_ai_signal': 'Hold',
                'current_price': None,
                'in_buy_position': False,
                'entry_price': None,
                'stop_loss_price': None,
                'high_since_entry': None,
                'last_notification_price': None
            }
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

            for ticker in STOCK_TICKERS:
                state = stock_states[ticker]
                model_info = loaded_models.get(ticker)
                
                stock_update_time = datetime.now().strftime('%H:%M:%S')

                if not model_info:
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: No AI model loaded. Skipping AI signal generation.")
                    continue

                df_live, fundamental_data, news_sentiment = fetch_data(ticker, LIVE_DATA_PERIOD, LIVE_DATA_INTERVAL)

                if df_live is None or df_live.empty:
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: Could not fetch recent data or data is insufficient.")
                    continue
                
                df_live_with_indicators = calculate_indicators(df_live)

                if df_live_with_indicators.empty or len(df_live_with_indicators) < MIN_REQUIRED_BARS_FOR_ANALYSIS:
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: Data too short ({len(df_live_with_indicators)} bars) after indicator calculation for analysis. Min required: {MIN_REQUIRED_BARS_FOR_ANALYSIS}.")
                    continue
                
                current_price_from_df = df_live_with_indicators['close'].iloc[-1]
                state['current_price'] = pd.to_numeric(current_price_from_df, errors='coerce')

                if pd.isna(state['current_price']):
                    logging.warning(f"{stock_update_time} ⚠️ {ticker}: Current price is NaN after conversion. Skipping further logic for this ticker.")
                    continue

                detected_patterns = detect_all_chart_patterns(df_live_with_indicators.copy(), MIN_REQUIRED_BARS_FOR_ANALYSIS)
                
                ai_signal = generate_ai_signal(df_live_with_indicators, model_info, ticker, fundamental_data, news_sentiment, detected_patterns)
                
                recent_high = df_live_with_indicators['high'].iloc[-min(20, len(df_live_with_indicators)):].max()
                recent_low = df_live_with_indicators['low'].iloc[-min(20, len(df_live_with_indicators)):].min()
                
                signal_int = {'BUY (AI)': 1, 'SELL (AI)': 0, 'Hold (AI)': 2, 'Hold (AI Model Error)': 2, 'Hold (Insufficient Data)': 2, 'Hold (Feature Mismatch)': 2, 'Hold (No Data)': 2}.get(ai_signal, 2)
                trade_details = generate_trade_signals(signal_int, state['current_price'], fundamental_data, news_sentiment, detected_patterns, recent_high, recent_low)
                
                price_changed_significantly = False
                if state['last_notification_price'] is not None and state['last_notification_price'] != 0:
                    price_change_percent = abs(state['current_price'] - state['last_notification_price']) / state['last_notification_price']
                    if price_change_percent > 0.005:
                        price_changed_significantly = True

                if state['in_buy_position']:
                    if state['high_since_entry'] is None or state['current_price'] > state['high_since_entry']:
                        state['high_since_entry'] = state['current_price']
                        new_trailing_sl = state['high_since_entry'] * (1 - TRAILING_STOP_LOSS_PERCENTAGE)
                        state['stop_loss_price'] = max(state['stop_loss_price'] if state['stop_loss_price'] is not None else 0, new_trailing_sl)
                        if VERBOSE_MODE:
                            logging.info(f"{stock_update_time} ✅ {ticker}: New high since entry. Trailing SL updated to {state['stop_loss_price']:.2f}")

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
                        continue

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
                        continue

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
                        continue
                    
                    else:
                        if ai_signal != state['last_ai_signal'] or price_changed_significantly:
                            p_l_percent = (state['current_price'] - state['entry_price']) / state['entry_price'] * 100 if state['entry_price'] != 0 else 0
                            notification_message = f"🟢 {ticker}: AI says {ai_signal}. Still in BUY position since {state['entry_price']:.2f}. Current: {state['current_price']:.2f} (SL: {state['stop_loss_price']:.2f}). Current P/L: {p_l_percent:.2f}%."
                            state['last_ai_signal'] = ai_signal
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

                else:
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
                            state['high_since_entry'] = entry_price
                            state['last_ai_signal'] = 'BUY (AI)'
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

                    elif ai_signal == 'SELL (AI)':
                        if ai_signal != state['last_ai_signal'] or price_changed_significantly:
                            notification_message = f"🔴 {ticker}: AI SELL SIGNAL. Not in position. Staying out for now. Current price: {state['current_price']:.2f}. Rationale: {'; '.join(trade_details['rationale'])}"
                            state['last_ai_signal'] = ai_signal
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

                    elif ai_signal == 'Hold (AI)':
                        if ai_signal != state['last_ai_signal'] or price_changed_significantly:
                            notification_message = f"🔵 {ticker}: AI says HOLD. Current price: {state['current_price']:.2f}. Not in position. Rationale: {'; '.join(trade_details['rationale'])}"
                            state['last_ai_signal'] = ai_signal
                            state['last_notification_price'] = state['current_price']
                            logging.info(f"{stock_update_time} {notification_message}")

            print_trade_log()
            save_output_to_json(stock_states, trade_log)
            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logging.info("\n\n--- Bot Interrupted by User (Ctrl+C) ---")
        print_trade_log()
        save_output_to_json(stock_states, trade_log)
        logging.info("Exiting monitoring mode.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n\n--- An unexpected error occurred: {e} ---")
        import traceback
        traceback.print_exc()
        print_trade_log()
        save_output_to_json(stock_states, trade_log)
        logging.info("Exiting monitoring mode due to error.")
        sys.exit(1)

