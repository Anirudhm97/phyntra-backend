

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# CRITICAL FIX 1: Add CORS for your frontend domain
CORS(app, origins=[
    'https://phyntra.ai',
    'https://www.phyntra.ai', 
    'http://localhost:3000'  # For local development
])

# CRITICAL FIX 2: Add GET method to health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Phyntra backend is running",
        "service": "phyntra-backend"
    })

# CRITICAL FIX 3: Add POST method to process-invoice endpoint
@app.route('/process-invoice', methods=['POST', 'OPTIONS'])
def process_invoice():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "status": "failed"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "status": "failed"
            }), 400
        
        # Basic file validation
        allowed_extensions = {'pdf', 'jpg', 'jpeg', 'png'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                "error": "File type not supported. Use PDF, JPG, or PNG",
                "status": "failed"
            }), 400
        
        # TODO: Add your actual invoice processing logic here
        # For now, return success response
        return jsonify({
            "status": "success",
            "message": f"Processing {file.filename}",
            "filename": file.filename
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

# OPTIONAL: Add a root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "Phyntra Backend API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process_invoice": "/process-invoice"
        }
    })

# CRITICAL FIX 4: Handle all CORS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
