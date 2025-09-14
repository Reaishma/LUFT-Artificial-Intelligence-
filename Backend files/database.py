from pymongo import MongoClient
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from bson import ObjectId

class DatabaseManager:
    def __init__(self):
        # Initialize MongoDB connection
        self.client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.db = self.client.ml_platform
        
        # Initialize collections
        self.datasets = self.db.datasets
        self.models = self.db.models
        self.projects = self.db.projects
        self.training_jobs = self.db.training_jobs
        self.deployments = self.db.deployments

    def save_dataset(self, name: str, file_path: str, file_type: str, metadata: Dict) -> str:
        """Save dataset information to database"""
        dataset_doc = {
            'name': name,
            'file_path': file_path,
            'file_type': file_type,
            'metadata': metadata,
            'created_at': datetime.utcnow(),
            'status': 'uploaded'
        }
        result = self.datasets.insert_one(dataset_doc)
        return str(result.inserted_id)

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID"""
        try:
            return self.datasets.find_one({'_id': ObjectId(dataset_id)})
        except:
            return None

    def list_datasets(self) -> List[Dict]:
        """List all datasets"""
        datasets = list(self.datasets.find().sort('created_at', -1))
        for dataset in datasets:
            dataset['_id'] = str(dataset['_id'])
        return datasets

    def save_model(self, name: str, algorithm: str, parameters: Dict, 
                   dataset_id: str, file_path: str = None) -> str:
        """Save trained model to database"""
        model_doc = {
            'name': name,
            'algorithm': algorithm,
            'parameters': parameters,
            'dataset_id': dataset_id,
            'file_path': file_path,
            'created_at': datetime.utcnow(),
            'status': 'created',
            'metrics': {}
        }
        result = self.models.insert_one(model_doc)
        return str(result.inserted_id)

    def update_model_status(self, model_id: str, status: str, metrics: Dict = None):
        """Update model training status and metrics"""
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        if metrics:
            update_data['metrics'] = metrics
            
        self.models.update_one(
            {'_id': ObjectId(model_id)},
            {'$set': update_data}
        )

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model by ID"""
        try:
            return self.models.find_one({'_id': ObjectId(model_id)})
        except:
            return None

    def list_models(self) -> List[Dict]:
        """List all models"""
        models = list(self.models.find().sort('created_at', -1))
        for model in models:
            model['_id'] = str(model['_id'])
        return models

    def save_training_job(self, model_id: str, status: str, progress: float = 0.0) -> str:
        """Save training job status"""
        job_doc = {
            'model_id': model_id,
            'status': status,
            'progress': progress,
            'created_at': datetime.utcnow(),
            'logs': []
        }
        result = self.training_jobs.insert_one(job_doc)
        return str(result.inserted_id)

    def update_training_job(self, job_id: str, status: str, progress: float, logs: List[str] = None):
        """Update training job progress"""
        update_data = {
            'status': status,
            'progress': progress,
            'updated_at': datetime.utcnow()
        }
        if logs:
            update_data['logs'] = logs
            
        self.training_jobs.update_one(
            {'_id': ObjectId(job_id)},
            {'$set': update_data}
        )

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """Get training job by ID"""
        try:
            return self.training_jobs.find_one({'_id': ObjectId(job_id)})
        except:
            return None