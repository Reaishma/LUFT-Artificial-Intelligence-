# Additional API endpoints for the ML Platform
import os
import uuid
import threading
import pandas as pd
from datetime import datetime
from flask import request, jsonify, send_file
from werkzeug.utils import secure_filename

def add_api_endpoints(app, db, ml_algorithms, UPLOAD_FOLDER, MODELS_FOLDER, allowed_file):
    """Add comprehensive API endpoints to the Flask app"""
    
    @app.route('/api/upload', methods=['POST'])
    def upload_data():
        """Upload dataset endpoint"""
        try:
            if 'file' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No file provided"
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "success": False,
                    "error": "No file selected"
                }), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                dataset_id = str(uuid.uuid4())
                file_path = os.path.join(UPLOAD_FOLDER, f"{dataset_id}_{filename}")
                file.save(file_path)
                
                # Analyze the uploaded data
                file_type = filename.rsplit('.', 1)[1].lower()
                try:
                    if file_type == 'csv':
                        df = pd.read_csv(file_path)
                    elif file_type in ['xlsx', 'xls']:
                        df = pd.read_excel(file_path)
                    elif file_type == 'json':
                        df = pd.read_json(file_path)
                    
                    # Create analysis metadata
                    analysis = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'missing_values': df.isnull().sum().to_dict(),
                        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
                        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
                    }
                    
                    # Save dataset to database
                    db.save_dataset(dataset_id, filename, file_path, file_type, analysis)
                    
                    return jsonify({
                        "success": True,
                        "message": "File uploaded successfully",
                        "dataset_id": dataset_id,
                        "filename": filename,
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    return jsonify({
                        "success": False,
                        "error": f"Failed to process file: {str(e)}"
                    }), 400
            
            return jsonify({
                "success": False,
                "error": "File type not allowed. Please upload CSV, Excel, or JSON files."
            }), 400
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/datasets/<dataset_id>/preview', methods=['GET'])
    def preview_dataset(dataset_id):
        """Preview dataset contents"""
        try:
            dataset = db.get_dataset(dataset_id)
            if not dataset:
                return jsonify({
                    "success": False,
                    "error": "Dataset not found"
                }), 404
            
            file_type = dataset['file_type']
            if file_type == 'csv':
                df = pd.read_csv(dataset['file_path'])
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(dataset['file_path'])
            elif file_type == 'json':
                df = pd.read_json(dataset['file_path'])
            
            # Return first 10 rows as preview
            preview_data = {
                "columns": list(df.columns),
                "data": df.head(10).to_dict('records'),
                "shape": df.shape,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            return jsonify({
                "success": True,
                "preview": preview_data
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/preprocess', methods=['POST'])
    def preprocess_data():
        """Preprocess dataset"""
        try:
            data = request.get_json()
            dataset_id = data.get('dataset_id')
            config = data.get('config', {})
            
            dataset = db.get_dataset(dataset_id)
            if not dataset:
                return jsonify({
                    "success": False,
                    "error": "Dataset not found"
                }), 404
            
            # Load data
            file_type = dataset['file_type']
            if file_type == 'csv':
                df = pd.read_csv(dataset['file_path'])
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(dataset['file_path'])
            elif file_type == 'json':
                df = pd.read_json(dataset['file_path'])
            
            # Preprocess data
            processed_df, preprocessing_info = ml_algorithms.preprocess_data(df, config)
            
            # Save preprocessed data
            processed_filename = f"preprocessed_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            processed_path = os.path.join(UPLOAD_FOLDER, processed_filename)
            processed_df.to_csv(processed_path, index=False)
            
            return jsonify({
                "success": True,
                "message": "Data preprocessed successfully",
                "preprocessing_info": preprocessing_info,
                "processed_path": processed_path
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/train', methods=['POST'])
    def train_model():
        """Train a machine learning model"""
        try:
            data = request.get_json()
            dataset_id = data.get('dataset_id')
            algorithm_id = data.get('algorithm')
            parameters = data.get('parameters', {})
            target_column = data.get('target_column')
            model_name = data.get('model_name', f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if not all([dataset_id, algorithm_id, target_column]):
                return jsonify({
                    "success": False,
                    "error": "Missing required parameters: dataset_id, algorithm, target_column"
                }), 400
            
            dataset = db.get_dataset(dataset_id)
            if not dataset:
                return jsonify({
                    "success": False,
                    "error": "Dataset not found"
                }), 404
            
            # Create model entry
            model_id = str(uuid.uuid4())
            db.save_model(model_id, model_name, algorithm_id, parameters, dataset_id)
            
            # Create training job
            job_id = str(uuid.uuid4())
            db.save_training_job(job_id, model_id)
            
            # Start training in background
            training_thread = threading.Thread(
                target=_train_model_background,
                args=(db, ml_algorithms, job_id, model_id, dataset, algorithm_id, parameters, target_column, MODELS_FOLDER)
            )
            training_thread.start()
            
            return jsonify({
                "success": True,
                "message": "Training started",
                "model_id": model_id,
                "job_id": job_id
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/training-jobs/<job_id>', methods=['GET'])
    def get_training_job(job_id):
        """Get training job status"""
        try:
            job = db.get_training_job(job_id)
            if not job:
                return jsonify({
                    "success": False,
                    "error": "Training job not found"
                }), 404
            
            return jsonify({
                "success": True,
                "job": job
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/models/<model_id>', methods=['GET'])
    def get_model_details(model_id):
        """Get specific model details"""
        try:
            model = db.get_model(model_id)
            if not model:
                return jsonify({
                    "success": False,
                    "error": "Model not found"
                }), 404
            
            return jsonify({
                "success": True,
                "model": model
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/models/<model_id>/predict', methods=['POST'])
    def make_prediction(model_id):
        """Make predictions with a trained model"""
        try:
            data = request.get_json()
            input_data = data.get('data')
            
            model = db.get_model(model_id)
            if not model or model['status'] != 'trained':
                return jsonify({
                    "success": False,
                    "error": "Trained model not found"
                }), 404
            
            # Load model and make prediction
            trained_model = ml_algorithms.load_model(model['file_path'])
            predictions = trained_model.predict([input_data])
            
            return jsonify({
                "success": True,
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/models/<model_id>/download', methods=['GET'])
    def download_model(model_id):
        """Download trained model file"""
        try:
            model = db.get_model(model_id)
            if not model or model['status'] != 'trained':
                return jsonify({
                    "success": False,
                    "error": "Trained model not found"
                }), 404
            
            return send_file(
                model['file_path'],
                as_attachment=True,
                download_name=f"{model['name']}.pkl"
            )
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


def _train_model_background(db, ml_algorithms, job_id, model_id, dataset, algorithm_id, parameters, target_column, models_folder):
    """Background training function"""
    try:
        # Update progress
        db.update_training_job(job_id, 'training', 20.0)
        
        # Load data
        file_type = dataset['file_type']
        if file_type == 'csv':
            df = pd.read_csv(dataset['file_path'])
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(dataset['file_path'])
        elif file_type == 'json':
            df = pd.read_json(dataset['file_path'])
        
        db.update_training_job(job_id, 'training', 40.0)
        
        # Prepare ML data
        ml_data = ml_algorithms.prepare_ml_data(df, target_column)
        
        db.update_training_job(job_id, 'training', 60.0)
        
        # Train model
        trained_model, training_info = ml_algorithms.train_model(
            algorithm_id, 
            ml_data['X_train'], 
            ml_data['y_train'],
            parameters
        )
        
        db.update_training_job(job_id, 'training', 80.0)
        
        # Evaluate model
        metrics = ml_algorithms.evaluate_model(
            trained_model, 
            ml_data['X_test'], 
            ml_data['y_test'],
            algorithm_id
        )
        
        # Save model
        model_filename = f"model_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(models_folder, model_filename)
        ml_algorithms.save_model(trained_model, model_path)
        
        # Update database
        db.update_model(model_id, status='trained', metrics=metrics, file_path=model_path)
        db.update_training_job(job_id, 'completed', 100.0)
        
    except Exception as e:
        db.update_training_job(job_id, 'failed', 0.0, [str(e)])
        db.update_model(model_id, status='failed', metrics={'error': str(e)})