import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, silhouette_score
import joblib
import os
from typing import Dict, Any, Tuple, List
from datetime import datetime

class MLModelManager:
    """
    Comprehensive machine learning model management system
    Supports multiple algorithms for classification, regression, clustering, and dimensionality reduction
    """
    
    def __init__(self):
        self.available_algorithms = {
            # Regression Algorithms
            'linear_regression': {
                'class': LinearRegression,
                'type': 'regression',
                'name': 'Linear Regression',
                'description': 'Simple linear approach to modeling relationships between variables',
                'default_params': {},
                'tunable_params': ['fit_intercept', 'normalize']
            },
            'decision_tree_regressor': {
                'class': DecisionTreeRegressor,
                'type': 'regression', 
                'name': 'Decision Tree Regressor',
                'description': 'Tree-based model for non-linear regression tasks',
                'default_params': {'random_state': 42, 'max_depth': 10},
                'tunable_params': ['max_depth', 'min_samples_split', 'min_samples_leaf']
            },
            'random_forest_regressor': {
                'class': RandomForestRegressor,
                'type': 'regression',
                'name': 'Random Forest Regressor',
                'description': 'Ensemble of decision trees for robust regression',
                'default_params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
                'tunable_params': ['n_estimators', 'max_depth', 'min_samples_split']
            },
            'svm_regressor': {
                'class': SVR,
                'type': 'regression',
                'name': 'Support Vector Regressor',
                'description': 'Support Vector Machine for regression tasks',
                'default_params': {'kernel': 'rbf', 'C': 1.0},
                'tunable_params': ['kernel', 'C', 'gamma', 'epsilon']
            },
            'neural_network_regressor': {
                'class': MLPRegressor,
                'type': 'regression',
                'name': 'Neural Network Regressor',
                'description': 'Multi-layer perceptron for complex non-linear regression',
                'default_params': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42},
                'tunable_params': ['hidden_layer_sizes', 'learning_rate', 'alpha']
            },
            
            # Classification Algorithms
            'logistic_regression': {
                'class': LogisticRegression,
                'type': 'classification',
                'name': 'Logistic Regression',
                'description': 'Linear model for binary and multiclass classification',
                'default_params': {'max_iter': 1000, 'random_state': 42},
                'tunable_params': ['C', 'penalty', 'solver']
            },
            'decision_tree_classifier': {
                'class': DecisionTreeClassifier,
                'type': 'classification',
                'name': 'Decision Tree Classifier',
                'description': 'Interpretable tree-based classification algorithm',
                'default_params': {'random_state': 42, 'max_depth': 10},
                'tunable_params': ['max_depth', 'min_samples_split', 'criterion']
            },
            'random_forest_classifier': {
                'class': RandomForestClassifier,
                'type': 'classification',
                'name': 'Random Forest Classifier',
                'description': 'Ensemble classifier with high accuracy and robustness',
                'default_params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
                'tunable_params': ['n_estimators', 'max_depth', 'min_samples_split']
            },
            'svm_classifier': {
                'class': SVC,
                'type': 'classification',
                'name': 'Support Vector Classifier',
                'description': 'Powerful classifier for high-dimensional data',
                'default_params': {'kernel': 'rbf', 'random_state': 42},
                'tunable_params': ['kernel', 'C', 'gamma']
            },
            'neural_network_classifier': {
                'class': MLPClassifier,
                'type': 'classification',
                'name': 'Neural Network Classifier',
                'description': 'Multi-layer perceptron for complex classification patterns',
                'default_params': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42},
                'tunable_params': ['hidden_layer_sizes', 'learning_rate', 'alpha']
            },
            
            # Clustering Algorithms
            'kmeans': {
                'class': KMeans,
                'type': 'clustering',
                'name': 'K-Means Clustering',
                'description': 'Partition data into k clusters based on feature similarity',
                'default_params': {'n_clusters': 3, 'random_state': 42, 'n_init': 10},
                'tunable_params': ['n_clusters', 'init', 'max_iter']
            },
            'hierarchical_clustering': {
                'class': AgglomerativeClustering,
                'type': 'clustering',
                'name': 'Hierarchical Clustering',
                'description': 'Build hierarchy of clusters using linkage criteria',
                'default_params': {'n_clusters': 3, 'linkage': 'ward'},
                'tunable_params': ['n_clusters', 'linkage', 'affinity']
            },
            
            # Dimensionality Reduction
            'pca': {
                'class': PCA,
                'type': 'dimensionality_reduction',
                'name': 'Principal Component Analysis',
                'description': 'Reduce dimensionality while preserving variance',
                'default_params': {'n_components': 2},
                'tunable_params': ['n_components', 'whiten']
            }
        }
        
        self.model_cache = {}
        self.training_history = []

    def get_available_algorithms(self) -> Dict[str, List[Dict]]:
        """Get categorized list of available algorithms with descriptions"""
        categorized = {
            'regression': [],
            'classification': [],
            'clustering': [],
            'dimensionality_reduction': []
        }
        
        for alg_id, alg_info in self.available_algorithms.items():
            categorized[alg_info['type']].append({
                'id': alg_id,
                'name': alg_info['name'],
                'type': alg_info['type'],
                'description': alg_info['description'],
                'tunable_params': alg_info['tunable_params']
            })
            
        return categorized

    def create_model(self, algorithm_id: str, parameters: Dict[str, Any] = None):
        """Create model instance with specified parameters"""
        if algorithm_id not in self.available_algorithms:
            raise ValueError(f"Algorithm {algorithm_id} not supported")
        
        alg_info = self.available_algorithms[algorithm_id]
        params = alg_info['default_params'].copy()
        
        if parameters:
            # Validate and update parameters
            valid_params = self._validate_parameters(algorithm_id, parameters)
            params.update(valid_params)
            
        return alg_info['class'](**params)

    def _validate_parameters(self, algorithm_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize model parameters"""
        alg_info = self.available_algorithms[algorithm_id]
        tunable_params = alg_info.get('tunable_params', [])
        
        valid_params = {}
        for param, value in parameters.items():
            if param in tunable_params or param in alg_info['default_params']:
                # Type conversion and validation
                if param in ['max_depth', 'n_estimators', 'n_clusters', 'n_components']:
                    valid_params[param] = max(1, int(value))
                elif param in ['C', 'gamma', 'alpha', 'learning_rate']:
                    valid_params[param] = max(0.0001, float(value))
                elif param in ['max_iter']:
                    valid_params[param] = max(100, int(value))
                else:
                    valid_params[param] = value
                    
        return valid_params

    def train_model(self, algorithm_id: str, X_train, y_train, parameters: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Train a model and return model instance with training info"""
        try:
            start_time = datetime.now()
            model = self.create_model(algorithm_id, parameters)
            
            alg_type = self.available_algorithms[algorithm_id]['type']
            
            # Handle different algorithm types
            if alg_type in ['clustering', 'dimensionality_reduction']:
                # Unsupervised learning
                if algorithm_id == 'pca':
                    model.fit(X_train)
                    transformed_data = model.transform(X_train)
                else:  # clustering
                    labels = model.fit_predict(X_train)
                    model.labels_ = labels
            else:
                # Supervised learning
                model.fit(X_train, y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Collect training information
            training_info = {
                'algorithm': algorithm_id,
                'algorithm_name': self.available_algorithms[algorithm_id]['name'],
                'training_time': training_time,
                'parameters': parameters or {},
                'feature_count': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'sample_count': X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train),
                'training_start': start_time.isoformat(),
                'training_end': end_time.isoformat(),
                'status': 'completed'
            }
            
            # Store in training history
            self.training_history.append(training_info.copy())
            
            return model, training_info
            
        except Exception as e:
            error_info = {
                'algorithm': algorithm_id,
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(error_info)
            raise Exception(f"Training failed for {algorithm_id}: {str(e)}")

    def evaluate_model(self, model, X_test, y_test, algorithm_id: str) -> Dict[str, Any]:
        """Comprehensive model evaluation with appropriate metrics"""
        try:
            metrics = {}
            alg_type = self.available_algorithms[algorithm_id]['type']
            
            if alg_type == 'clustering':
                # Clustering evaluation
                if hasattr(model, 'labels_'):
                    predictions = model.labels_[:len(X_test)] if len(model.labels_) >= len(X_test) else model.predict(X_test)
                else:
                    predictions = model.fit_predict(X_test)
                
                # Calculate clustering metrics
                if len(set(predictions)) > 1:
                    metrics['silhouette_score'] = float(silhouette_score(X_test, predictions))
                else:
                    metrics['silhouette_score'] = 0.0
                    
                metrics['n_clusters'] = len(set(predictions))
                metrics['cluster_sizes'] = [int(np.sum(predictions == i)) for i in set(predictions)]
                
            elif alg_type == 'dimensionality_reduction':
                # PCA evaluation
                if algorithm_id == 'pca':
                    metrics['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
                    metrics['cumulative_variance'] = np.cumsum(model.explained_variance_ratio_).tolist()
                    metrics['n_components'] = model.n_components_
                    metrics['total_variance_explained'] = float(np.sum(model.explained_variance_ratio_))
                    
            elif alg_type == 'classification':
                # Classification metrics
                predictions = model.predict(X_test)
                probabilities = None
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X_test)
                    except:
                        pass
                
                metrics['accuracy'] = float(accuracy_score(y_test, predictions))
                metrics['precision'] = float(precision_score(y_test, predictions, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_test, predictions, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_test, predictions, average='weighted', zero_division=0))
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, predictions)
                metrics['confusion_matrix'] = cm.tolist()
                
                # Class distribution
                unique_classes = np.unique(np.concatenate([y_test, predictions]))
                metrics['classes'] = unique_classes.tolist()
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    metrics['feature_importance'] = model.feature_importances_.tolist()
                
            elif alg_type == 'regression':
                # Regression metrics
                predictions = model.predict(X_test)
                metrics['r2_score'] = float(r2_score(y_test, predictions))
                metrics['mse'] = float(mean_squared_error(y_test, predictions))
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
                
                # Mean Absolute Error
                from sklearn.metrics import mean_absolute_error
                metrics['mae'] = float(mean_absolute_error(y_test, predictions))
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    metrics['feature_importance'] = model.feature_importances_.tolist()
                
            # Add general metrics
            metrics['evaluation_timestamp'] = datetime.now().isoformat()
            metrics['test_samples'] = len(X_test)
            
            return metrics
            
        except Exception as e:
            return {
                'error': f"Evaluation failed: {str(e)}",
                'evaluation_timestamp': datetime.now().isoformat()
            }

    def save_model(self, model, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """Save model to file with metadata"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare model package
            model_package = {
                'model': model,
                'metadata': metadata or {},
                'saved_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save using joblib
            joblib.dump(model_package, file_path)
            
            # Cache model locally
            model_id = os.path.basename(file_path).replace('.pkl', '')
            self.model_cache[model_id] = model_package
            
            return file_path
            
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def load_model(self, file_path: str):
        """Load model from file"""
        try:
            model_package = joblib.load(file_path)
            
            # Handle both new format (with metadata) and old format (just model)
            if isinstance(model_package, dict) and 'model' in model_package:
                return model_package['model'], model_package.get('metadata', {})
            else:
                return model_package, {}
                
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def get_model_info(self, algorithm_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific algorithm"""
        if algorithm_id not in self.available_algorithms:
            raise ValueError(f"Algorithm {algorithm_id} not found")
            
        alg_info = self.available_algorithms[algorithm_id].copy()
        
        # Add usage statistics if available
        usage_count = sum(1 for history in self.training_history if history.get('algorithm') == algorithm_id)
        success_count = sum(1 for history in self.training_history 
                          if history.get('algorithm') == algorithm_id and history.get('status') == 'completed')
        
        alg_info['usage_statistics'] = {
            'total_trainings': usage_count,
            'successful_trainings': success_count,
            'success_rate': success_count / usage_count if usage_count > 0 else 0
        }
        
        return alg_info

    def get_training_history(self, algorithm_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get training history, optionally filtered by algorithm"""
        history = self.training_history
        
        if algorithm_id:
            history = [h for h in history if h.get('algorithm') == algorithm_id]
            
        # Sort by timestamp (newest first) and limit
        history.sort(key=lambda x: x.get('timestamp', x.get('training_start', '')), reverse=True)
        return history[:limit]

    def clear_cache(self):
        """Clear model cache to free memory"""
        self.model_cache.clear()

    def get_recommendations(self, problem_type: str, dataset_size: int, feature_count: int) -> List[str]:
        """Get algorithm recommendations based on problem characteristics"""
        recommendations = []
        
        if problem_type == 'classification':
            if dataset_size < 1000:
                recommendations = ['logistic_regression', 'decision_tree_classifier']
            elif dataset_size < 10000:
                recommendations = ['random_forest_classifier', 'svm_classifier']
            else:
                recommendations = ['random_forest_classifier', 'neural_network_classifier']
                
        elif problem_type == 'regression':
            if dataset_size < 1000:
                recommendations = ['linear_regression', 'decision_tree_regressor']
            elif dataset_size < 10000:
                recommendations = ['random_forest_regressor', 'svm_regressor']
            else:
                recommendations = ['random_forest_regressor', 'neural_network_regressor']
                
        elif problem_type == 'clustering':
            if dataset_size < 5000:
                recommendations = ['kmeans', 'hierarchical_clustering']
            else:
                recommendations = ['kmeans']
                
        return recommendations