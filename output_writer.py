# output_writer.py
import json
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from config import WEB_UI_OUTPUT_PATH, VERBOSE_MODE

def convert_numpy_types(obj):
    """Recursively converts NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif pd.isna(obj):
        return None
    return obj

def save_output_to_json(stock_states: dict, trade_log: list):
    """
    Saves the current stock states and trade log to a JSON file for the web UI.
    """
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_states': stock_states,
        'trade_log': trade_log
    }
    
    output_data_serializable = convert_numpy_types(output_data)

    output_dir = os.path.dirname(WEB_UI_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    try:
        with open(WEB_UI_OUTPUT_PATH, 'w') as f:
            json.dump(output_data_serializable, f, indent=4)
        if VERBOSE_MODE:
            logging.info(f"Output saved to {WEB_UI_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Error saving output to {WEB_UI_OUTPUT_PATH}: {e}")

