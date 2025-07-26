import nest_asyncio

# Apply the patch for nested asyncio event loops.
nest_asyncio.apply()

from flask import Flask, request, jsonify
from main import AutonomousLangGraphSystem
import os
import logging
import threading

# Configure logging to make it visible in Vercel's logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# --- Lazy Initialization Setup ---
# We will initialize the system on the first request to avoid Vercel's cold start timeout.
# A lock is used to ensure thread-safe initialization.
system = None
init_error = None
system_lock = threading.Lock()

def initialize_system():
    """Initializes the AutonomousLangGraphSystem if it hasn't been already."""
    global system
    global init_error
    # Use a lock to ensure that this heavy initialization only runs once,
    # even if multiple requests come in simultaneously on a cold start.
    with system_lock:
        if system is None and init_error is None:
            try:
                logging.info("First request received. Initializing AutonomousLangGraphSystem...")
                system = AutonomousLangGraphSystem()
                logging.info("System initialized successfully.")
            except Exception as e:
                init_error = str(e)
                logging.error(f"FATAL: System could not be initialized during first request. Error: {init_error}", exc_info=True)

# --- API Routes ---

@app.route('/', methods=['GET'])
def home():
    """A simple health check endpoint."""
    # This endpoint does not trigger initialization.
    return "Agentic AI System is running. Use the /process endpoint to submit a query.", 200


@app.route('/process', methods=['POST'])
def process_query():
    """Main endpoint to process a user's query."""
    # Trigger the one-time initialization if it hasn't happened yet.
    if system is None:
        initialize_system()

    # Handle the case where the system failed to start up.
    if init_error:
        return jsonify({"error": f"System is not available: {init_error}"}), 500
    if not system:
        return jsonify({"error": "System initialization is in progress or has failed silently. Please try again."}), 503

    # --- Input Validation ---
    if not request.is_json:
        logging.warning("Request received without JSON body.")
        return jsonify({"error": "Invalid request: body must be JSON."}), 400

    data = request.get_json()
    query = data.get('query')

    if not query or not isinstance(query, str):
        logging.warning(f"Invalid query received: {query}")
        return jsonify({"error": "Invalid request: 'query' must be a non-empty string."}), 400

    logging.info(f"Processing query: '{query[:50]}...'")

    # --- Core Logic Execution ---
    try:
        result = system.process_query(query)
        logging.info("Query processed successfully.")
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"An error occurred during processing query '{query}': {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing the request."}), 500

# The 'if __name__ == "__main__":' block is for local testing and is ignored by Vercel.
if __name__ == "__main__":
    app.run(debug=True, port=5001)
