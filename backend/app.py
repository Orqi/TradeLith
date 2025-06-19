# backend/app.py
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS # Used for cross-origin requests
import subprocess
import os
import sys
import logging
import signal # For graceful shutdown

# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='../web-ui', static_url_path='/')
CORS(app) # Enable CORS for all routes

# Define the path to your main.py script
# Assuming app.py is in 'backend/' and main.py is in the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BOT_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'main.py')
WEB_UI_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'web-ui', 'output.json')

# Dictionary to keep track of running bot processes (to avoid multiple instances)
running_bot_processes = {}

@app.route('/')
def serve_index():
    """Serve the main index.html file."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.) from the web-ui folder."""
    return send_from_directory(app.static_folder, path)

@app.route('/api/train', methods=['POST'])
def train_bot_endpoint():
    """API endpoint to trigger the bot training."""
    logging.info("Received request to train bot.")
    if 'train_bot' in running_bot_processes and running_bot_processes['train_bot'].poll() is None:
        return jsonify({"status": "error", "message": "Bot training is already running."}), 409

    try:
        # Start the training process in the background
        # Use sys.executable to ensure the correct python interpreter is used
        process = subprocess.Popen(
            [sys.executable, BOT_SCRIPT_PATH, 'train'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Line-buffered output
        )
        running_bot_processes['train_bot'] = process
        logging.info("Bot training process started.")
        # For long-running processes, you might want to return immediately
        # and let the frontend poll for status/logs.
        return jsonify({"status": "success", "message": "Bot training initiated. Check backend server logs for progress."}), 200
    except Exception as e:
        logging.error(f"Error starting training process: {e}")
        return jsonify({"status": "error", "message": f"Failed to initiate training: {str(e)}"}), 500

@app.route('/api/monitor', methods=['POST'])
def monitor_bot_endpoint():
    """API endpoint to trigger the bot monitoring."""
    logging.info("Received request to monitor bot.")
    if 'monitor_bot' in running_bot_processes and running_bot_processes['monitor_bot'].poll() is None:
        return jsonify({"status": "error", "message": "Bot monitoring is already running."}), 409

    try:
        # Start the monitoring process in the background
        # It's crucial for monitoring to run as a separate process
        # because it's a long-running, blocking loop in main.py
        process = subprocess.Popen(
            [sys.executable, BOT_SCRIPT_PATH, 'monitor'],
            stdout=subprocess.PIPE, # Capture stdout for logging or future display
            stderr=subprocess.PIPE, # Capture stderr
            text=True,
            bufsize=1 # Line-buffered output
        )
        running_bot_processes['monitor_bot'] = process
        logging.info("Bot monitoring process started.")
        return jsonify({"status": "success", "message": "Bot monitoring initiated. Dashboard will start updating shortly."}), 200
    except Exception as e:
        logging.error(f"Error starting monitoring process: {e}")
        return jsonify({"status": "error", "message": f"Failed to initiate monitoring: {str(e)}"}), 500

@app.route('/api/stop_monitor', methods=['POST'])
def stop_monitor_endpoint():
    """API endpoint to stop the bot monitoring."""
    logging.info("Received request to stop monitor.")
    monitor_process = running_bot_processes.get('monitor_bot')
    if monitor_process and monitor_process.poll() is None: # Process is still running
        try:
            # Send SIGINT to gracefully stop the process (mimics Ctrl+C)
            monitor_process.send_signal(signal.SIGINT)
            monitor_process.wait(timeout=5) # Wait for it to terminate
            del running_bot_processes['monitor_bot']
            logging.info("Bot monitoring process stopped.")
            return jsonify({"status": "success", "message": "Bot monitoring stopped."}), 200
        except subprocess.TimeoutExpired:
            monitor_process.kill() # Force kill if it doesn't stop gracefully
            del running_bot_processes['monitor_bot']
            logging.warning("Bot monitoring process killed due to timeout.")
            return jsonify({"status": "warning", "message": "Bot monitoring force-stopped (did not terminate gracefully)."}), 200
        except Exception as e:
            logging.error(f"Error stopping monitoring process: {e}")
            return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"}), 500
    else:
        return jsonify({"status": "info", "message": "Bot monitoring is not currently running."}), 200


@app.route('/api/output.json', methods=['GET'])
def get_output_json():
    """API endpoint to serve the output.json file."""
    if not os.path.exists(WEB_UI_OUTPUT_PATH):
        return jsonify({"status": "error", "message": "output.json not found. Bot might not be running or training yet."}), 404
    return send_from_directory(os.path.dirname(WEB_UI_OUTPUT_PATH), os.path.basename(WEB_UI_OUTPUT_PATH))

if __name__ == '__main__':
    # Add project root to sys.path so modules can be imported
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    app.run(debug=True, port=5000) # Run on port 5000 by default. Set debug=False for production.
