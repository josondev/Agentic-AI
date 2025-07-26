from flask import Flask, request, jsonify
from main import AutonomousLangGraphSystem
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize your agentic system
# This assumes your main.py is structured in a way that this class can be imported.
# You might need to adjust main.py to avoid running the interactive loop when imported.
# A simple way is to put the `if __name__ == "__main__":` block at the end of main.py
try:
    system = AutonomousLangGraphSystem()
except Exception as e:
    # If initialization fails (e.g., missing API keys), we can handle it gracefully.
    system = None
    init_error = str(e)

@app.route('/', methods=['GET'])
def home():
    return "Agentic AI System is running. Use the /process endpoint to submit a query.", 200

@app.route('/process', methods=['POST'])
def process_query():
    if not system:
        return jsonify({"error": f"System not initialized: {init_error}"}), 500

    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query not provided"}), 400

    try:
        # The process_query method from your main.py script
        result = system.process_query(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

# This allows Vercel to run the app
if __name__ == "__main__":
    app.run(debug=True)
