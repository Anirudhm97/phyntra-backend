# Update your app.py CORS configuration

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# CRITICAL: Update CORS to include www.phyntra.ai
CORS(app, 
     origins=[
         'https://phyntra.ai',
         'https://www.phyntra.ai',  # This is the one being blocked!
         'http://localhost:3000'
     ],
     allow_headers=['Content-Type', 'Authorization'],
     allow_methods=['GET', 'POST', 'OPTIONS']
)

# Alternative: More permissive CORS (temporary fix)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Your existing routes...
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/process-invoice', methods=['POST', 'OPTIONS'])
def process_invoice():
    if request.method == 'OPTIONS':
        return '', 200
    
    # Your processing logic here
    return jsonify({"status": "success", "message": "Processing started"})
