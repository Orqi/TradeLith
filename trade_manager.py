# trade_manager.py
import pandas as pd
from datetime import datetime
import logging
from config import (
    VERBOSE_MODE, INITIAL_STOP_LOSS_PERCENTAGE, TRAILING_STOP_LOSS_PERCENTAGE,
    TAKE_PROFIT_PERCENTAGE, LIVE_DATA_INTERVAL, trade_log # Import global trade_log
)

# NOTE: trade_log is imported as a global variable from config.py.
# In a production system, this would be backed by a persistent database.

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

    if fundamental_data:
        if fundamental_data.get('pe_ratio') is not None and not pd.isna(fundamental_data['pe_ratio']):
            trade_info['rationale'].append(f"PE Ratio: {fundamental_data['pe_ratio']:.2f}")
        if fundamental_data.get('market_capitalization') is not None and not pd.isna(fundamental_data['market_capitalization']):
            trade_info['rationale'].append(f"Market Cap: {fundamental_data['market_capitalization']/1e9:.2f}B")
        if fundamental_data.get('book_value_per_share') is not None and not pd.isna(fundamental_data['book_value_per_share']):
            trade_info['rationale'].append(f"Book Value/Share: {fundamental_data['book_value_per_share']:.2f}")
        if fundamental_data.get('dividend_yield') is not None and not pd.isna(fundamental_data['dividend_yield']):
            trade_info['rationale'].append(f"Dividend Yield: {fundamental_data['dividend_yield']*100:.2f}%")

    if news_sentiment and news_sentiment.get('avg_news_sentiment') is not None and not pd.isna(news_sentiment['avg_news_sentiment']):
        sentiment = news_sentiment['avg_news_sentiment']
        if sentiment > 0.35:
            trade_info['rationale'].append(f"Strong News Sentiment ({sentiment:.2f})")
        elif sentiment < -0.35:
            trade_info['rationale'].append(f"Weak News Sentiment ({sentiment:.2f})")
        else:
            trade_info['rationale'].append(f"Neutral News Sentiment ({sentiment:.2f})")

    detected_patterns_list = [k for k, v in chart_patterns.items() if v]
    if detected_patterns_list:
        trade_info['rationale'].append(f"Detected Chart Patterns: {', '.join(detected_patterns_list)}")

    sl_pct = INITIAL_STOP_LOSS_PERCENTAGE
    tp1_rr_ratio = 1.5
    tp2_rr_ratio = 3.0

    if signal == 1:
        trade_info['rationale'].insert(0, "AI BUY SIGNAL")
        if recent_low > 0 and current_price - recent_low > 0:
            trade_info['stop_loss'] = max(recent_low * 0.99, current_price * (1 - sl_pct))
        else:
            trade_info['stop_loss'] = current_price * (1 - sl_pct)
        trade_info['stop_loss'] = min(trade_info['stop_loss'], current_price * 0.99)

        risk_amount = current_price - trade_info['stop_loss']
        if risk_amount <= 0:
            risk_amount = current_price * sl_pct
            trade_info['stop_loss'] = current_price - risk_amount

        trade_info['take_profit_1'] = current_price + (risk_amount * tp1_rr_ratio)
        trade_info['take_profit_2'] = current_price + (risk_amount * tp2_rr_ratio)
        trade_info['rationale'].append(f"Anticipated duration: Short-medium term (based on {LIVE_DATA_INTERVAL} data)")

    elif signal == 0:
        trade_info['rationale'].insert(0, "AI SELL SIGNAL")
        if recent_high > 0 and recent_high - current_price > 0:
            trade_info['stop_loss'] = min(recent_high * 1.01, current_price * (1 + sl_pct))
        else:
            trade_info['stop_loss'] = current_price * (1 + sl_pct)
        trade_info['stop_loss'] = max(trade_info['stop_loss'], current_price * 1.01)

        risk_amount = trade_info['stop_loss'] - current_price
        if risk_amount <= 0:
            risk_amount = current_price * sl_pct
            trade_info['stop_loss'] = current_price + risk_amount

        trade_info['take_profit_1'] = current_price - (risk_amount * tp1_rr_ratio)
        trade_info['take_profit_2'] = current_price - (risk_amount * tp2_rr_ratio)

        trade_info['take_profit_1'] = max(0.01, trade_info['take_profit_1'])
        trade_info['take_profit_2'] = max(0.01, trade_info['take_profit_2'])
        trade_info['rationale'].append(f"Anticipated duration: Short-medium term (based on {LIVE_DATA_INTERVAL} data)")

    else:
        trade_info['rationale'].insert(0, "AI HOLD SIGNAL")
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
        'details': trade_details
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
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        logging.info("\n" + df_log.to_string())

    closed_buys = df_log[df_log['type'] == 'EXIT']
    if not closed_buys.empty:
        total_profit_loss_percent = closed_buys['p_l_percent'].sum()
        logging.info(f"\nTotal P/L for closed positions: {total_profit_loss_percent:.2f}%")
    else:
        logging.info("\nNo closed positions to calculate total P/L.")

