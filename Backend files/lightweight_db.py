import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class LightweightDB:
    def __init__(self, db_path: str = 'ml_platform.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'uploaded'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                parameters TEXT,
                dataset_id TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'created',
                metrics TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                model_id TEXT,
                status TEXT DEFAULT 'started',
                progress REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                logs TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_dataset(self, dataset_id: str, name: str, file_path: str, file_type: str, metadata: Dict) -> str:
        """Save dataset information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO datasets (id, name, file_path, file_type, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (dataset_id, name, file_path, file_type, json.dumps(metadata)))
        
        conn.commit()
        conn.close()
        return dataset_id

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'file_path': row[2],
                'file_type': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'created_at': row[5],
                'status': row[6]
            }
        return None

    def list_datasets(self) -> List[Dict]:
        """List all datasets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM datasets ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        datasets = []
        for row in rows:
            datasets.append({
                'id': row[0],
                'name': row[1],
                'file_path': row[2],
                'file_type': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'created_at': row[5],
                'status': row[6]
            })
        return datasets

    def save_model(self, model_id: str, name: str, algorithm: str, parameters: Dict, dataset_id: str, file_path: str = None) -> str:
        """Save model information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO models (id, name, algorithm, parameters, dataset_id, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_id, name, algorithm, json.dumps(parameters), dataset_id, file_path))
        
        conn.commit()
        conn.close()
        return model_id

    def update_model(self, model_id: str, status: str = None, metrics: Dict = None, file_path: str = None):
        """Update model information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if status:
            updates.append('status = ?')
            values.append(status)
        if metrics:
            updates.append('metrics = ?')
            values.append(json.dumps(metrics))
        if file_path:
            updates.append('file_path = ?')
            values.append(file_path)
        
        if updates:
            values.append(model_id)
            cursor.execute(f'UPDATE models SET {", ".join(updates)} WHERE id = ?', values)
            conn.commit()
        
        conn.close()

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM models WHERE id = ?', (model_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'algorithm': row[2],
                'parameters': json.loads(row[3]) if row[3] else {},
                'dataset_id': row[4],
                'file_path': row[5],
                'created_at': row[6],
                'status': row[7],
                'metrics': json.loads(row[8]) if row[8] else {}
            }
        return None

    def list_models(self) -> List[Dict]:
        """List all models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM models ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        models = []
        for row in rows:
            models.append({
                'id': row[0],
                'name': row[1],
                'algorithm': row[2],
                'parameters': json.loads(row[3]) if row[3] else {},
                'dataset_id': row[4],
                'file_path': row[5],
                'created_at': row[6],
                'status': row[7],
                'metrics': json.loads(row[8]) if row[8] else {}
            })
        return models

    def save_training_job(self, job_id: str, model_id: str, status: str = 'started') -> str:
        """Save training job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_jobs (id, model_id, status)
            VALUES (?, ?, ?)
        ''', (job_id, model_id, status))
        
        conn.commit()
        conn.close()
        return job_id

    def update_training_job(self, job_id: str, status: str, progress: float, logs: List[str] = None):
        """Update training job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        log_data = json.dumps(logs) if logs else None
        cursor.execute('''
            UPDATE training_jobs 
            SET status = ?, progress = ?, logs = ?
            WHERE id = ?
        ''', (status, progress, log_data, job_id))
        
        conn.commit()
        conn.close()

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """Get training job by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM training_jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'model_id': row[1],
                'status': row[2],
                'progress': row[3],
                'created_at': row[4],
                'logs': json.loads(row[5]) if row[5] else []
            }
        return None