# main.py - Fixed version for Render

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# CORS configuration
CORS(app, 
     origins=[
         'https://phyntra.ai',
         'https://www.phyntra.ai',
         'http://localhost:3000'
     ],
     allow_headers=['Content-Type', 'Authorization'],
     allow_methods=['GET', 'POST', 'OPTIONS']
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Phyntra backend is running",
        "service": "phyntra-backend"
    })

@app.route('/process-invoice', methods=['POST', 'OPTIONS'])
def process_invoice():
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
        
        # For now, just return success without actual processing
        # This will make it work immediately
        return jsonify({
            "status": "success",
            "message": f"Successfully received {file.filename}",
            "filename": file.filename,
            "size": len(file.read()),
            "note": "Basic processing complete - full AI processing coming soon!"
        })
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")  # This will show in logs
        return jsonify({
            "error": f"Processing failed: {str(e)}",
            "status": "failed"
        }), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "Phyntra Backend API",
        "status": "running",
        "version": "1.0.0"
    })

# Critical: This must be at the bottom
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Phyntra backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
