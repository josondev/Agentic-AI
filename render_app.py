import nest_asyncio
nest_asyncio.apply()

from flask import Flask, request, jsonify
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Global system instance
system = None
init_error = None

def initialize_system():
    """Initialize the system on first request"""
    global system, init_error
    
    if system is None and init_error is None:
        try:
            logging.info("Initializing AutonomousLangGraphSystem...")
            
            # Disable Ollama for cloud deployment
            os.environ['OLLAMA_AVAILABLE'] = 'False'
            
            from main import AutonomousLangGraphSystem
            system = AutonomousLangGraphSystem()
            
            logging.info("✅ System initialized successfully")
        except Exception as e:
            init_error = str(e)
            logging.error(f"❌ System initialization failed: {init_error}", exc_info=True)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "Agentic AI System",
        "version": "1.0",
        "endpoints": {
            "health": "/",
            "process": "/process (POST)"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    if system is None:
        initialize_system()
    
    return jsonify({
        "status": "healthy" if system else "initializing",
        "system_ready": system is not None,
        "error": init_error
    }), 200 if system else 503

@app.route('/process', methods=['POST'])
def process_query():
    """Main endpoint to process queries"""
    # Initialize system on first request
    if system is None:
        initialize_system()
    
    # Check for initialization errors
    if init_error:
        return jsonify({
            "error": f"System initialization failed: {init_error}"
        }), 500
    
    if not system:
        return jsonify({
            "error": "System is still initializing. Please try again in a moment."
        }), 503
    
    # Validate request
    if not request.is_json:
        return jsonify({
            "error": "Request must be JSON"
        }), 400
    
    data = request.get_json()
    query = data.get('query')
    
    if not query or not isinstance(query, str):
        return jsonify({
            "error": "'query' must be a non-empty string"
        }), 400
    
    # Process the query
    logging.info(f"Processing query: '{query[:100]}...'")
    
    try:
        result = system.process_query(query)
        logging.info("✅ Query processed successfully")
        
        return jsonify({
            "success": True,
            "query": query,
            "answer": result.get("answer"),
            "thinking_log": result.get("thinking_log", [])
        }), 200
        
    except Exception as e:
        logging.error(f"❌ Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "An error occurred while processing your request",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    # Render sets the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
