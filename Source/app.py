import nest_asyncio

# Apply the patch before importing any other asyncio-related libraries.
# This is the crucial fix for the FUNCTION_INVOCATION_FAILED error on Vercel.
nest_asyncio.apply()

from flask import Flask, request, jsonify
from main import AutonomousLangGraphSystem
import os
import logging

# Configure logging to make it visible in Vercel's logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# --- System Initialization ---
# Initialize your agentic system once when the application starts.
# This is more efficient than creating it on every request.
system = None
init_error = None
try:
    logging.info("Initializing AutonomousLangGraphSystem...")
    system = AutonomousLangGraphSystem()
    logging.info("System initialized successfully.")
except Exception as e:
    # If initialization fails (e.g., missing API keys), log the error
    # and store it to return in API responses.
    init_error = str(e)
    logging.error(f"FATAL: System could not be initialized. Error: {init_error}")

# --- API Routes ---

@app.route('/', methods=['GET'])
def home():
    """A simple health check endpoint to confirm the service is running."""
    if system:
        return "Agentic AI System is running. Use the /process endpoint to submit a query.", 200
    else:
        return f"Agentic AI System failed to initialize. Error: {init_error}", 500


@app.route('/process', methods=['POST'])
def process_query():
    """Main endpoint to process a user's query."""
    # Handle the case where the system failed to start up.
    if not system:
        return jsonify({"error": f"System not initialized: {init_error}"}), 500

    # --- Input Validation ---
    # This helps prevent errors from malformed requests.
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
    # This block handles errors during the execution of your main agent logic.
    # This is crucial for preventing Vercel's FUNCTION_INVOCATION_FAILED error.
    try:
        # The process_query method from your main.py script
        result = system.process_query(query)
        logging.info("Query processed successfully.")
        return jsonify(result), 200
    except Exception as e:
        # If any unexpected error occurs in your system, log it and return a
        # generic 500 Internal Server Error to the client.
        logging.error(f"An error occurred during processing query '{query}': {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing the request."}), 500

# This allows Vercel to run the app.
# The 'if __name__ == "__main__":' block is for local testing and is ignored by Vercel.
if __name__ == "__main__":
    # For local development, you can run this script directly.
    # The debug=True flag provides helpful error pages but should be False in production.
    app.run(debug=True, port=5001)
