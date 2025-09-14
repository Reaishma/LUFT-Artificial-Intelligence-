import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# Import our custom modules
from models import MLModelManager
from data import DataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Advanced model training system with background processing, progress tracking,
    hyperparameter optimization, and comprehensive experiment management
    """
    
    def __init__(self, models_dir: str = 'trained_models', logs_dir: str = 'training_logs'):
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.model_manager = MLModelManager()
        self.data_processor = DataProcessor()
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Training state management
        self.active_jobs = {}
        self.completed_jobs = {}
        self.training_history = []
        self.job_lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {}
        self.experiment_configs = {}

    def start_training_job(self, config: Dict[str, Any]) -> str:
        """Start a new training job in the background"""
        job_id = str(uuid.uuid4())
        
        # Validate configuration
        required_fields = ['dataset_path', 'algorithm', 'target_column']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Initialize job
        job_info = {
            'job_id': job_id,
            'status': 'started',
            'progress': 0.0,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'logs': [],
            'model_path': None,
            'metrics': {},
            'error': None
        }
        
        with self.job_lock:
            self.active_jobs[job_id] = job_info
        
        # Start training in background thread
        thread = threading.Thread(
            target=self._train_model_background,
            args=(job_id, config),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started training job {job_id} for algorithm {config['algorithm']}")
        return job_id

    def _train_model_background(self, job_id: str, config: Dict[str, Any]):
        """Background training process"""
        try:
            self._update_job_status(job_id, 'initializing', 5.0, ['Initializing training process...'])
            
            # Load and analyze data
            self._update_job_status(job_id, 'loading_data', 10.0, ['Loading dataset...'])
            df = self.data_processor.load_data(config['dataset_path'])
            
            self._update_job_status(job_id, 'analyzing_data', 20.0, ['Analyzing data quality...'])
            data_analysis = self.data_processor.analyze_data(df, config['target_column'])
            
            # Data preprocessing
            self._update_job_status(job_id, 'preprocessing', 30.0, ['Preprocessing data...'])
            preprocessing_config = config.get('preprocessing', {})
            preprocessed_df, preprocessing_info = self.data_processor.preprocess_data(df, preprocessing_config)
            
            # Prepare ML data
            self._update_job_status(job_id, 'preparing_ml_data', 40.0, ['Preparing ML data splits...'])
            ml_data = self.data_processor.prepare_ml_data(
                preprocessed_df, 
                config['target_column'],
                test_size=config.get('test_size', 0.2)
            )
            
            # Model training
            self._update_job_status(job_id, 'training', 50.0, ['Training model...'])
            algorithm = config['algorithm']
            parameters = config.get('parameters', {})
            
            trained_model, training_info = self.model_manager.train_model(
                algorithm,
                ml_data['X_train'],
                ml_data['y_train'],
                parameters
            )
            
            # Model evaluation
            self._update_job_status(job_id, 'evaluating', 80.0, ['Evaluating model performance...'])
            metrics = self.model_manager.evaluate_model(
                trained_model,
                ml_data['X_test'],
                ml_data['y_test'],
                algorithm
            )
            
            # Save model
            self._update_job_status(job_id, 'saving', 90.0, ['Saving trained model...'])
            model_filename = f"{job_id}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            
            model_metadata = {
                'job_id': job_id,
                'algorithm': algorithm,
                'training_info': training_info,
                'metrics': metrics,
                'data_analysis': data_analysis,
                'preprocessing_info': preprocessing_info,
                'ml_data_info': {
                    'feature_names': ml_data['feature_names'],
                    'target_name': ml_data['target_name'],
                    'train_size': ml_data['train_size'],
                    'test_size': ml_data['test_size']
                },
                'config': config
            }
            
            self.model_manager.save_model(trained_model, model_path, model_metadata)
            
            # Complete job
            completion_logs = [
                f'Model training completed successfully!',
                f'Model saved to: {model_path}',
                f'Training time: {training_info.get("training_time", 0):.2f} seconds',
                f'Final metrics: {metrics}'
            ]
            
            self._complete_job(job_id, 'completed', 100.0, completion_logs, model_path, metrics)
            
            # Save experiment results
            self._save_experiment_results(job_id, config, metrics, training_info)
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            self._complete_job(job_id, 'failed', 0.0, [error_msg], None, {}, str(e))

    def _update_job_status(self, job_id: str, status: str, progress: float, logs: List[str]):
        """Update job status and progress"""
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job['status'] = status
                job['progress'] = progress
                job['logs'].extend(logs)
                job['last_update'] = datetime.now().isoformat()
                
                # Log progress
                for log in logs:
                    logger.info(f"Job {job_id}: {log}")

    def _complete_job(self, job_id: str, status: str, progress: float, logs: List[str], 
                     model_path: Optional[str], metrics: Dict[str, Any], error: Optional[str] = None):
        """Complete a training job"""
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job['status'] = status
                job['progress'] = progress
                job['logs'].extend(logs)
                job['model_path'] = model_path
                job['metrics'] = metrics
                job['error'] = error
                job['end_time'] = datetime.now().isoformat()
                
                # Move to completed jobs
                self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
                
                # Add to training history
                self.training_history.append(job.copy())
                
                logger.info(f"Job {job_id} completed with status: {status}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a training job"""
        with self.job_lock:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id].copy()
            elif job_id in self.completed_jobs:
                return self.completed_jobs[job_id].copy()
            return None

    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active training jobs"""
        with self.job_lock:
            return list(self.active_jobs.values())

    def list_completed_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List completed training jobs"""
        with self.job_lock:
            # Sort by completion time, newest first
            sorted_jobs = sorted(
                self.completed_jobs.values(),
                key=lambda x: x.get('end_time', ''),
                reverse=True
            )
            return sorted_jobs[:limit]

    def stop_job(self, job_id: str) -> bool:
        """Stop an active training job"""
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job['status'] = 'stopping'
                job['logs'].append('Training job stop requested...')
                logger.info(f"Stop requested for job {job_id}")
                return True
            return False

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        with self.job_lock:
            total_jobs = len(self.training_history)
            completed_jobs = len([j for j in self.training_history if j.get('status') == 'completed'])
            failed_jobs = len([j for j in self.training_history if j.get('status') == 'failed'])
            active_jobs = len(self.active_jobs)
            
            # Algorithm popularity
            algorithm_counts = {}
            algorithm_success_rates = {}
            
            for job in self.training_history:
                algorithm = job.get('config', {}).get('algorithm', 'unknown')
                algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
                
                if algorithm not in algorithm_success_rates:
                    algorithm_success_rates[algorithm] = {'total': 0, 'successful': 0}
                
                algorithm_success_rates[algorithm]['total'] += 1
                if job.get('status') == 'completed':
                    algorithm_success_rates[algorithm]['successful'] += 1
            
            # Calculate success rates
            for algorithm in algorithm_success_rates:
                stats = algorithm_success_rates[algorithm]
                stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            
            return {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'active_jobs': active_jobs,
                'success_rate': completed_jobs / total_jobs if total_jobs > 0 else 0,
                'algorithm_popularity': algorithm_counts,
                'algorithm_success_rates': algorithm_success_rates,
                'last_updated': datetime.now().isoformat()
            }

    def _save_experiment_results(self, job_id: str, config: Dict[str, Any], 
                                metrics: Dict[str, Any], training_info: Dict[str, Any]):
        """Save experiment results for analysis"""
        experiment_data = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'training_info': training_info,
            'performance_summary': self._create_performance_summary(metrics)
        }
        
        # Save to experiments log
        experiment_file = os.path.join(self.logs_dir, f'experiment_{job_id}.json')
        with open(experiment_file, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        # Update performance tracking
        algorithm = config['algorithm']
        if algorithm not in self.performance_metrics:
            self.performance_metrics[algorithm] = []
        
        self.performance_metrics[algorithm].append({
            'job_id': job_id,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    def _create_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a performance summary from metrics"""
        summary = {}
        
        # Classification metrics
        if 'accuracy' in metrics:
            summary['primary_metric'] = 'accuracy'
            summary['primary_score'] = metrics['accuracy']
        elif 'f1_score' in metrics:
            summary['primary_metric'] = 'f1_score'
            summary['primary_score'] = metrics['f1_score']
        
        # Regression metrics
        elif 'r2_score' in metrics:
            summary['primary_metric'] = 'r2_score'
            summary['primary_score'] = metrics['r2_score']
        elif 'mse' in metrics:
            summary['primary_metric'] = 'mse'
            summary['primary_score'] = metrics['mse']
        
        # Clustering metrics
        elif 'silhouette_score' in metrics:
            summary['primary_metric'] = 'silhouette_score'
            summary['primary_score'] = metrics['silhouette_score']
        
        # Add performance tier
        if 'primary_score' in summary:
            score = summary['primary_score']
            if summary['primary_metric'] in ['accuracy', 'f1_score', 'r2_score', 'silhouette_score']:
                if score >= 0.9:
                    summary['performance_tier'] = 'excellent'
                elif score >= 0.8:
                    summary['performance_tier'] = 'good'
                elif score >= 0.7:
                    summary['performance_tier'] = 'fair'
                else:
                    summary['performance_tier'] = 'poor'
            else:  # For metrics where lower is better (like MSE)
                summary['performance_tier'] = 'unknown'
        
        return summary

    def get_best_models(self, algorithm: str = None, metric: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get best performing models based on metrics"""
        candidates = []
        
        # Collect all completed jobs with metrics
        for job in self.training_history:
            if job.get('status') != 'completed' or not job.get('metrics'):
                continue
            
            job_algorithm = job.get('config', {}).get('algorithm')
            if algorithm and job_algorithm != algorithm:
                continue
            
            metrics = job.get('metrics', {})
            
            # Determine primary metric if not specified
            if not metric:
                if 'accuracy' in metrics:
                    metric = 'accuracy'
                elif 'f1_score' in metrics:
                    metric = 'f1_score'
                elif 'r2_score' in metrics:
                    metric = 'r2_score'
                elif 'silhouette_score' in metrics:
                    metric = 'silhouette_score'
                else:
                    continue
            
            if metric in metrics:
                candidates.append({
                    'job_id': job['job_id'],
                    'algorithm': job_algorithm,
                    'metric_name': metric,
                    'metric_value': metrics[metric],
                    'model_path': job.get('model_path'),
                    'timestamp': job.get('end_time'),
                    'all_metrics': metrics
                })
        
        # Sort by metric value (descending for most metrics)
        reverse_sort = metric not in ['mse', 'rmse', 'mae']  # These are better when lower
        candidates.sort(key=lambda x: x['metric_value'], reverse=reverse_sort)
        
        return candidates[:limit]

    def run_hyperparameter_optimization(self, base_config: Dict[str, Any], 
                                       param_grid: Dict[str, List[Any]], 
                                       max_trials: int = 10) -> str:
        """Run hyperparameter optimization"""
        optimization_id = str(uuid.uuid4())
        logger.info(f"Starting hyperparameter optimization {optimization_id}")
        
        # Generate parameter combinations
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Limit combinations
        if len(param_combinations) > max_trials:
            import random
            param_combinations = random.sample(param_combinations, max_trials)
        
        # Start optimization jobs
        optimization_jobs = []
        
        for i, param_combo in enumerate(param_combinations):
            # Create config for this trial
            trial_config = base_config.copy()
            trial_parameters = dict(zip(param_names, param_combo))
            trial_config['parameters'] = trial_parameters
            trial_config['optimization_id'] = optimization_id
            trial_config['trial_number'] = i + 1
            
            # Start training job
            job_id = self.start_training_job(trial_config)
            optimization_jobs.append(job_id)
            
            logger.info(f"Started optimization trial {i+1}/{len(param_combinations)}: {trial_parameters}")
        
        # Save optimization info
        optimization_info = {
            'optimization_id': optimization_id,
            'base_config': base_config,
            'param_grid': param_grid,
            'job_ids': optimization_jobs,
            'total_trials': len(param_combinations),
            'start_time': datetime.now().isoformat()
        }
        
        optimization_file = os.path.join(self.logs_dir, f'optimization_{optimization_id}.json')
        with open(optimization_file, 'w') as f:
            json.dump(optimization_info, f, indent=2, default=str)
        
        return optimization_id

    def export_training_report(self, output_path: str):
        """Export comprehensive training report"""
        report = {
            'report_generated': datetime.now().isoformat(),
            'statistics': self.get_training_statistics(),
            'training_history': self.training_history[-50:],  # Last 50 jobs
            'performance_metrics': self.performance_metrics,
            'best_models': self.get_best_models(limit=20),
            'active_jobs': self.list_active_jobs(),
            'system_info': {
                'models_directory': self.models_dir,
                'logs_directory': self.logs_dir,
                'total_experiments': len(self.training_history)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report exported to {output_path}")

def main():
    """Command line interface for model training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Platform Model Trainer')
    parser.add_argument('--config', type=str, required=True, help='Training configuration JSON file')
    parser.add_argument('--models-dir', type=str, default='trained_models', help='Models directory')
    parser.add_argument('--logs-dir', type=str, default='training_logs', help='Logs directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(args.models_dir, args.logs_dir)
    
    # Start training
    job_id = trainer.start_training_job(config)
    print(f"Started training job: {job_id}")
    
    # Monitor progress
    import time
    while True:
        status = trainer.get_job_status(job_id)
        if not status:
            break
            
        print(f"Status: {status['status']}, Progress: {status['progress']:.1f}%")
        
        if status['status'] in ['completed', 'failed']:
            if status['status'] == 'completed':
                print(f"Training completed! Model saved to: {status.get('model_path')}")
                print(f"Metrics: {status.get('metrics')}")
            else:
                print(f"Training failed: {status.get('error')}")
            break
            
        time.sleep(5)

if __name__ == '__main__':
    main()