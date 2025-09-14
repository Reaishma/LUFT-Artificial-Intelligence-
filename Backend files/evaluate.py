import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, mean_squared_error, r2_score,
    silhouette_score, adjusted_rand_score
)
import joblib
import logging

# Import our custom modules
from models import MLModelManager
from data import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation system with advanced metrics, 
    visualizations, and performance analysis capabilities
    """
    
    def __init__(self, models_dir: str = 'trained_models', results_dir: str = 'evaluation_results'):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.model_manager = MLModelManager()
        self.data_processor = DataProcessor()
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Evaluation history
        self.evaluation_history = []
        
    def evaluate_model(self, model_path: str, test_data_path: str = None, 
                      target_column: str = None, detailed: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with detailed metrics and analysis
        """
        try:
            # Load model
            model, metadata = self.model_manager.load_model(model_path)
            
            # Extract information from metadata
            algorithm_id = metadata.get('algorithm', 'unknown')
            ml_data_info = metadata.get('ml_data_info', {})
            original_target = ml_data_info.get('target_name', target_column)
            
            if not test_data_path:
                # Use original test data if available
                test_data_path = metadata.get('config', {}).get('dataset_path')
                if not test_data_path:
                    raise ValueError("No test data provided and no original dataset path found")
            
            # Load and prepare test data
            df = self.data_processor.load_data(test_data_path)
            
            if not target_column and not original_target:
                raise ValueError("Target column must be specified")
            
            target_col = target_column or original_target
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in test data")
            
            # Prepare data using same preprocessing as training
            preprocessing_info = metadata.get('preprocessing_info', {})
            if preprocessing_info:
                # Apply same preprocessing steps
                df, _ = self._apply_preprocessing(df, preprocessing_info)
            
            # Split features and target
            X_test = df.drop(columns=[target_col])
            y_test = df[target_col]
            
            # Ensure feature alignment with training data
            expected_features = ml_data_info.get('feature_names', [])
            if expected_features:
                # Align features with training data
                X_test = self._align_features(X_test, expected_features)
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Get probabilities if available (for classification)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_test)
                except:
                    pass
            
            # Determine problem type
            algorithm_info = self.model_manager.get_model_info(algorithm_id)
            problem_type = algorithm_info['type']
            
            # Calculate metrics based on problem type
            evaluation_results = {
                'model_path': model_path,
                'algorithm': algorithm_id,
                'problem_type': problem_type,
                'test_samples': len(X_test),
                'evaluation_timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            if problem_type == 'classification':
                evaluation_results.update(
                    self._evaluate_classification(y_test, predictions, probabilities, detailed)
                )
            elif problem_type == 'regression':
                evaluation_results.update(
                    self._evaluate_regression(y_test, predictions, detailed)
                )
            elif problem_type == 'clustering':
                evaluation_results.update(
                    self._evaluate_clustering(X_test, predictions, detailed)
                )
            
            # Additional analysis if detailed
            if detailed:
                evaluation_results.update(
                    self._detailed_analysis(X_test, y_test, predictions, problem_type, algorithm_id)
                )
            
            # Save evaluation results
            self._save_evaluation_results(evaluation_results)
            
            # Add to history
            self.evaluation_history.append(evaluation_results)
            
            logger.info(f"Model evaluation completed for {model_path}")
            return evaluation_results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(error_msg)
            return {
                'error': error_msg,
                'model_path': model_path,
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def _evaluate_classification(self, y_true, y_pred, y_proba=None, detailed=True) -> Dict[str, Any]:
        """Comprehensive classification evaluation"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
        )
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        results['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        results['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        results['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Additional metrics
        results['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        results['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        results['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Class-specific metrics
        unique_classes = sorted(list(set(y_true) | set(y_pred)))
        results['classes'] = unique_classes
        results['n_classes'] = len(unique_classes)
        
        # Per-class metrics
        if len(unique_classes) > 2:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            results['per_class_metrics'] = {
                str(cls): {
                    'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                    'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                    'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
                }
                for i, cls in enumerate(unique_classes)
            }
        
        # ROC AUC and PR AUC (for binary and multiclass)
        if y_proba is not None:
            try:
                if len(unique_classes) == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    results['roc_auc'] = float(auc(fpr, tpr))
                    results['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    }
                    
                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                    results['pr_auc'] = float(auc(recall, precision))
                    results['pr_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist()
                    }
                else:
                    # Multiclass - compute macro-averaged ROC AUC
                    from sklearn.metrics import roc_auc_score
                    results['roc_auc_macro'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
                    results['roc_auc_weighted'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
            except Exception as e:
                logger.warning(f"Could not compute ROC/PR curves: {e}")
        
        # Classification report
        if detailed:
            report = classification_report(y_true, y_pred, output_dict=True)
            results['classification_report'] = report
        
        return results
    
    def _evaluate_regression(self, y_true, y_pred, detailed=True) -> Dict[str, Any]:
        """Comprehensive regression evaluation"""
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, median_absolute_error,
            explained_variance_score, max_error
        )
        
        results = {}
        
        # Basic metrics
        results['r2_score'] = float(r2_score(y_true, y_pred))
        results['mse'] = float(mean_squared_error(y_true, y_pred))
        results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        results['mae'] = float(mean_absolute_error(y_true, y_pred))
        results['median_ae'] = float(median_absolute_error(y_true, y_pred))
        
        # Additional metrics
        results['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        results['max_error'] = float(max_error(y_true, y_pred))
        
        # Percentage errors
        results['mape'] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else None
        
        # Residuals analysis
        residuals = y_true - y_pred
        results['residuals_stats'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q75': float(np.percentile(residuals, 75))
        }
        
        if detailed:
            # Prediction vs actual correlation
            results['pred_actual_correlation'] = float(np.corrcoef(y_true, y_pred)[0, 1])
            
            # Distribution comparison
            results['actual_stats'] = {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true))
            }
            results['predicted_stats'] = {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            }
        
        return results
    
    def _evaluate_clustering(self, X, cluster_labels, detailed=True) -> Dict[str, Any]:
        """Comprehensive clustering evaluation"""
        results = {}
        
        # Basic metrics
        n_clusters = len(np.unique(cluster_labels))
        results['n_clusters'] = n_clusters
        
        if n_clusters > 1:
            results['silhouette_score'] = float(silhouette_score(X, cluster_labels))
            
            # Cluster sizes
            unique, counts = np.unique(cluster_labels, return_counts=True)
            results['cluster_sizes'] = dict(zip(unique.astype(int).tolist(), counts.tolist()))
            
            # Inertia (if available)
            try:
                from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
                results['calinski_harabasz_score'] = float(calinski_harabasz_score(X, cluster_labels))
                results['davies_bouldin_score'] = float(davies_bouldin_score(X, cluster_labels))
            except Exception as e:
                logger.warning(f"Could not compute additional clustering metrics: {e}")
        
        if detailed:
            # Cluster centroids analysis
            results['cluster_analysis'] = {}
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = X[cluster_mask]
                
                results['cluster_analysis'][int(cluster_id)] = {
                    'size': int(np.sum(cluster_mask)),
                    'centroid': np.mean(cluster_data, axis=0).tolist() if len(cluster_data) > 0 else [],
                    'std': np.std(cluster_data, axis=0).tolist() if len(cluster_data) > 0 else []
                }
        
        return results
    
    def _detailed_analysis(self, X_test, y_test, predictions, problem_type, algorithm_id) -> Dict[str, Any]:
        """Additional detailed analysis"""
        analysis = {}
        
        # Feature importance (if available from model)
        # This would need the actual model object, which we'd need to modify the function signature for
        
        # Prediction confidence analysis
        if hasattr(predictions, 'std'):
            analysis['prediction_uncertainty'] = {
                'mean_std': float(np.mean(np.std(predictions, axis=0))),
                'max_std': float(np.max(np.std(predictions, axis=0)))
            }
        
        # Data distribution analysis
        if problem_type in ['classification', 'regression']:
            # Actual vs predicted distribution comparison
            analysis['distribution_analysis'] = {
                'actual_distribution': self._get_distribution_stats(y_test),
                'predicted_distribution': self._get_distribution_stats(predictions)
            }
        
        return analysis
    
    def _get_distribution_stats(self, data) -> Dict[str, Any]:
        """Get distribution statistics for data"""
        if pd.api.types.is_numeric_dtype(data):
            return {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'percentiles': {
                    'p25': float(np.percentile(data, 25)),
                    'p50': float(np.percentile(data, 50)),
                    'p75': float(np.percentile(data, 75)),
                    'p90': float(np.percentile(data, 90))
                }
            }
        else:
            # Categorical data
            unique, counts = np.unique(data, return_counts=True)
            return {
                'unique_count': len(unique),
                'most_frequent': str(unique[np.argmax(counts)]),
                'least_frequent': str(unique[np.argmin(counts)]),
                'value_counts': dict(zip([str(u) for u in unique], counts.tolist()))
            }
    
    def _apply_preprocessing(self, df: pd.DataFrame, preprocessing_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply same preprocessing steps as used during training"""
        # This is a simplified version - in practice, you'd want to save and reload
        # the actual preprocessing transformers used during training
        
        preprocessing_config = preprocessing_info.get('preprocessing_config', {})
        processed_df, info = self.data_processor.preprocess_data(df, preprocessing_config)
        
        return processed_df, info
    
    def _align_features(self, X_test: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
        """Align test features with training features"""
        # Add missing columns with zeros
        for feature in expected_features:
            if feature not in X_test.columns:
                X_test[feature] = 0
        
        # Remove extra columns and reorder
        X_test = X_test[expected_features]
        
        return X_test
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(results['model_path']).replace('.pkl', '')
        filename = f"evaluation_{model_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def compare_models(self, model_paths: List[str], test_data_path: str, 
                      target_column: str, metric: str = 'auto') -> Dict[str, Any]:
        """Compare multiple models on the same test dataset"""
        comparison_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'test_data_path': test_data_path,
            'target_column': target_column,
            'models': [],
            'ranking': [],
            'summary': {}
        }
        
        # Evaluate each model
        for model_path in model_paths:
            logger.info(f"Evaluating model: {model_path}")
            results = self.evaluate_model(model_path, test_data_path, target_column, detailed=False)
            
            if 'error' not in results:
                comparison_results['models'].append(results)
        
        if not comparison_results['models']:
            comparison_results['error'] = "No models could be evaluated successfully"
            return comparison_results
        
        # Determine comparison metric
        if metric == 'auto':
            first_model = comparison_results['models'][0]
            problem_type = first_model.get('problem_type')
            
            if problem_type == 'classification':
                metric = 'f1_score'
            elif problem_type == 'regression':
                metric = 'r2_score'
            elif problem_type == 'clustering':
                metric = 'silhouette_score'
            else:
                metric = 'accuracy'
        
        # Rank models by chosen metric
        models_with_metric = []
        for model in comparison_results['models']:
            if metric in model:
                models_with_metric.append({
                    'model_path': model['model_path'],
                    'algorithm': model['algorithm'],
                    'metric_value': model[metric],
                    'all_metrics': model
                })
        
        # Sort by metric (descending for most metrics, ascending for error metrics)
        reverse_sort = metric not in ['mse', 'rmse', 'mae']
        models_with_metric.sort(key=lambda x: x['metric_value'], reverse=reverse_sort)
        
        comparison_results['ranking'] = models_with_metric
        comparison_results['best_model'] = models_with_metric[0] if models_with_metric else None
        comparison_results['comparison_metric'] = metric
        
        # Summary statistics
        if models_with_metric:
            metric_values = [m['metric_value'] for m in models_with_metric]
            comparison_results['summary'] = {
                'total_models': len(models_with_metric),
                'metric_used': metric,
                'best_score': metric_values[0],
                'worst_score': metric_values[-1],
                'mean_score': float(np.mean(metric_values)),
                'std_score': float(np.std(metric_values))
            }
        
        # Save comparison results
        self._save_comparison_results(comparison_results)
        
        return comparison_results
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save model comparison results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_comparison_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Model comparison results saved to {filepath}")
    
    def generate_evaluation_report(self, model_path: str, output_format: str = 'json') -> str:
        """Generate comprehensive evaluation report"""
        # This would generate visualizations and detailed reports
        # For now, returning the path where report would be saved
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(model_path).replace('.pkl', '')
        report_filename = f"evaluation_report_{model_name}_{timestamp}.{output_format}"
        report_path = os.path.join(self.results_dir, report_filename)
        
        logger.info(f"Evaluation report would be generated at: {report_path}")
        return report_path
    
    def get_evaluation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        return self.evaluation_history[-limit:]

def main():
    """Command line interface for model evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Platform Model Evaluator')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--detailed', action='store_true', help='Detailed evaluation')
    parser.add_argument('--results-dir', type=str, default='evaluation_results', help='Results directory')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(results_dir=args.results_dir)
    
    # Evaluate model
    results = evaluator.evaluate_model(
        args.model, 
        args.test_data, 
        args.target, 
        args.detailed
    )
    
    if 'error' in results:
        print(f"Evaluation failed: {results['error']}")
        return
    
    # Print results
    print(f"Model Evaluation Results for {args.model}")
    print(f"Algorithm: {results['algorithm']}")
    print(f"Problem Type: {results['problem_type']}")
    print(f"Test Samples: {results['test_samples']}")
    
    if results['problem_type'] == 'classification':
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
    elif results['problem_type'] == 'regression':
        print(f"R² Score: {results['r2_score']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
    elif results['problem_type'] == 'clustering':
        print(f"Silhouette Score: {results['silhouette_score']:.4f}")
        print(f"Number of Clusters: {results['n_clusters']}")

if __name__ == '__main__':
    main()