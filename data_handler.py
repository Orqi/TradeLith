import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import logging
from config import ALPHAVANTAGE_API_KEY, VERBOSE_MODE

def fetch_data(ticker: str, period: str, interval: str) -> tuple[pd.DataFrame | None, dict, dict]:

    logging.info(f"  [Data Fetch] Fetching data for {ticker} (yfinance)...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        logging.warning(f"  [Data Fetch] No data returned for {ticker} from yfinance for period='{period}', interval='{interval}'.")
        return None, {}, {}

    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            cleaned_col = str(col[0]).replace(' ', '_').lower()
        else:
            cleaned_col = str(col).replace(' ', '_').lower()
        new_columns.append(cleaned_col)
    df.columns = new_columns

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"  [Data Fetch Error] Missing one or more essential OHLCV columns for {ticker}. Found: {df.columns.tolist()}")
        return None, {}, {}
    
    if 'volume' not in df.columns or df['volume'].isnull().all():
        df['volume'] = 0 

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    if df.empty:
        logging.warning(f"  [Data Fetch] DataFrame is empty after cleaning OHLC NaNs for {ticker}.")
        return None, {}, {}

    fundamental_data = {}
    news_sentiment_data = {}

    if ALPHAVANTAGE_API_KEY and ALPHAVANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        try:
            logging.info(f"  [Data Fetch] Fetching fundamental data for {ticker} (Alpha Vantage OVERVIEW)...")
            overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker.split('.')[0]}&apikey={ALPHAVANTAGE_API_KEY}"
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

            logging.info(f"  [Data Fetch] Fetching news sentiment for {ticker} (Alpha Vantage NEWS_SENTIMENT)...")
            news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker.split('.')[0]}&sort=LATEST&limit=5&apikey={ALPHAVANTAGE_API_KEY}"
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
                    news_sentiment_data['avg_news_sentiment'] = 0.0
            else:
                news_sentiment_data['avg_news_sentiment'] = 0.0
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
