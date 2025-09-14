from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import threading
from werkzeug.utils import secure_filename

# Import our lightweight components
from storage.lightweight_db import LightweightDB
from ml.lightweight_algorithms import LightweightMLAlgorithms
from api_endpoints import add_api_endpoints

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'trained_models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Initialize components
db = LightweightDB()
ml_algorithms = LightweightMLAlgorithms()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add comprehensive API endpoints
add_api_endpoints(app, db, ml_algorithms, UPLOAD_FOLDER, MODELS_FOLDER, allowed_file)

@app.route('/')
def landing_page():
    """Landing page with platform introduction and benefits"""
    return render_template('landing.html')

@app.route('/model-builder')
def model_builder():
    """Model builder interface"""
    return render_template('model_builder.html')

@app.route('/model-gallery')
def model_gallery():
    """Model gallery showcasing pre-built models"""
    return render_template('model_gallery.html')

@app.route('/data-uploader')
def data_uploader():
    """Data upload interface"""
    return render_template('data_uploader.html')

@app.route('/model-trainer')
def model_trainer():
    """Model training interface"""
    return render_template('model_trainer.html')

@app.route('/model-evaluator')
def model_evaluator():
    """Model evaluation interface"""
    return render_template('model_evaluator.html')

@app.route('/model-deployer')
def model_deployer():
    """Model deployment interface"""
    return render_template('model_deployer.html')

@app.route('/documentation')
def documentation():
    """Documentation section"""
    return render_template('documentation.html')

@app.route('/support')
def support():
    """Support section"""
    return render_template('support.html')

# API endpoints
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "ML Platform API is running",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Get available ML algorithms"""
    try:
        algorithms = ml_algorithms.get_available_algorithms()
        return jsonify({
            "success": True,
            "algorithms": algorithms
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all trained models"""
    try:
        models = db.list_models()
        return jsonify({
            "success": True,
            "models": models
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get all datasets"""
    try:
        datasets = db.list_datasets()
        return jsonify({
            "success": True,
            "datasets": datasets
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Cache control for development
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
