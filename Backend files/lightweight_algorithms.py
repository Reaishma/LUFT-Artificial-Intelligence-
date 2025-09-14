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

class LightweightMLAlgorithms:
    def __init__(self):
        self.available_algorithms = {
            # Regression
            'linear_regression': {
                'class': LinearRegression,
                'type': 'regression',
                'name': 'Linear Regression',
                'default_params': {}
            },
            'decision_tree_regressor': {
                'class': DecisionTreeRegressor,
                'type': 'regression', 
                'name': 'Decision Tree Regressor',
                'default_params': {'random_state': 42, 'max_depth': 10}
            },
            'random_forest_regressor': {
                'class': RandomForestRegressor,
                'type': 'regression',
                'name': 'Random Forest Regressor',
                'default_params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10}
            },
            'svm_regressor': {
                'class': SVR,
                'type': 'regression',
                'name': 'Support Vector Regressor',
                'default_params': {'kernel': 'rbf'}
            },
            'neural_network_regressor': {
                'class': MLPRegressor,
                'type': 'regression',
                'name': 'Neural Network Regressor',
                'default_params': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42}
            },
            # Classification
            'logistic_regression': {
                'class': LogisticRegression,
                'type': 'classification',
                'name': 'Logistic Regression',
                'default_params': {'max_iter': 1000, 'random_state': 42}
            },
            'decision_tree_classifier': {
                'class': DecisionTreeClassifier,
                'type': 'classification',
                'name': 'Decision Tree Classifier',
                'default_params': {'random_state': 42, 'max_depth': 10}
            },
            'random_forest_classifier': {
                'class': RandomForestClassifier,
                'type': 'classification',
                'name': 'Random Forest Classifier',
                'default_params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10}
            },
            'svm_classifier': {
                'class': SVC,
                'type': 'classification',
                'name': 'Support Vector Classifier',
                'default_params': {'kernel': 'rbf', 'random_state': 42}
            },
            'neural_network_classifier': {
                'class': MLPClassifier,
                'type': 'classification',
                'name': 'Neural Network Classifier',
                'default_params': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42}
            },
            # Clustering
            'kmeans': {
                'class': KMeans,
                'type': 'clustering',
                'name': 'K-Means Clustering',
                'default_params': {'n_clusters': 3, 'random_state': 42, 'n_init': 10}
            },
            'hierarchical_clustering': {
                'class': AgglomerativeClustering,
                'type': 'clustering',
                'name': 'Hierarchical Clustering',
                'default_params': {'n_clusters': 3}
            },
            # Dimensionality Reduction
            'pca': {
                'class': PCA,
                'type': 'dimensionality_reduction',
                'name': 'Principal Component Analysis',
                'default_params': {'n_components': 2}
            }
        }

    def get_available_algorithms(self) -> Dict[str, List[Dict]]:
        """Get categorized list of available algorithms"""
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
                'type': alg_info['type']
            })
            
        return categorized

    def create_model(self, algorithm_id: str, parameters: Dict[str, Any] = None):
        """Create model instance"""
        if algorithm_id not in self.available_algorithms:
            raise ValueError(f"Algorithm {algorithm_id} not supported")
        
        alg_info = self.available_algorithms[algorithm_id]
        params = alg_info['default_params'].copy()
        
        if parameters:
            params.update(parameters)
            
        return alg_info['class'](**params)

    def train_model(self, algorithm_id: str, X_train, y_train, parameters: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Train a model"""
        try:
            start_time = datetime.now()
            model = self.create_model(algorithm_id, parameters)
            
            # Handle different algorithm types
            if self.available_algorithms[algorithm_id]['type'] in ['clustering', 'dimensionality_reduction']:
                # Unsupervised learning
                if algorithm_id == 'pca':
                    model.fit(X_train)
                else:  # clustering
                    labels = model.fit_predict(X_train)
                    model.labels_ = labels
            else:
                # Supervised learning
                model.fit(X_train, y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            training_info = {
                'algorithm': algorithm_id,
                'training_time': training_time,
                'parameters': parameters or {},
                'feature_count': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'sample_count': X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train),
                'status': 'completed'
            }
            
            return model, training_info
            
        except Exception as e:
            raise Exception(f"Training failed for {algorithm_id}: {str(e)}")

    def evaluate_model(self, model, X_test, y_test, algorithm_id: str) -> Dict[str, Any]:
        """Evaluate trained model"""
        try:
            metrics = {}
            alg_type = self.available_algorithms[algorithm_id]['type']
            
            if alg_type == 'clustering':
                # Clustering evaluation
                if hasattr(model, 'labels_'):
                    predictions = model.labels_[:len(X_test)] if len(model.labels_) >= len(X_test) else model.predict(X_test)
                else:
                    predictions = model.fit_predict(X_test)
                
                if len(set(predictions)) > 1:
                    metrics['silhouette_score'] = float(silhouette_score(X_test, predictions))
                else:
                    metrics['silhouette_score'] = 0.0
                metrics['n_clusters'] = len(set(predictions))
                
            elif alg_type == 'dimensionality_reduction':
                # PCA evaluation
                if algorithm_id == 'pca':
                    metrics['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
                    metrics['cumulative_variance'] = np.cumsum(model.explained_variance_ratio_).tolist()
                    metrics['n_components'] = model.n_components_
                    
            elif alg_type == 'classification':
                # Classification metrics
                predictions = model.predict(X_test)
                metrics['accuracy'] = float(accuracy_score(y_test, predictions))
                metrics['precision'] = float(precision_score(y_test, predictions, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_test, predictions, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_test, predictions, average='weighted', zero_division=0))
                
            elif alg_type == 'regression':
                # Regression metrics
                predictions = model.predict(X_test)
                metrics['r2_score'] = float(r2_score(y_test, predictions))
                metrics['mse'] = float(mean_squared_error(y_test, predictions))
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
                
            return metrics
            
        except Exception as e:
            return {'error': f"Evaluation failed: {str(e)}"}

    def save_model(self, model, file_path: str) -> str:
        """Save model to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(model, file_path)
            return file_path
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def load_model(self, file_path: str):
        """Load model from file"""
        try:
            return joblib.load(file_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def preprocess_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess data for ML"""
        processed_df = df.copy()
        preprocessing_info = {'steps_applied': []}
        
        # Handle missing values
        if config.get('handle_missing', True):
            strategy = config.get('missing_strategy', 'mean')
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0:
                if strategy == 'mean':
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                elif strategy == 'median':
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
                else:
                    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)
            
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if len(processed_df[col].mode()) > 0 else 'Unknown')
                    
            preprocessing_info['steps_applied'].append('missing_values_handled')

        # Encode categorical variables
        if config.get('encode_categorical', True):
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                processed_df[col + '_encoded'] = le.fit_transform(processed_df[col].astype(str))
                processed_df = processed_df.drop(columns=[col])
            preprocessing_info['steps_applied'].append('categorical_encoded')

        # Scale features
        if config.get('scale_features', True):
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = StandardScaler() if config.get('scaling_method', 'standard') == 'standard' else MinMaxScaler()
                processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            preprocessing_info['steps_applied'].append('features_scaled')

        return processed_df, preprocessing_info

    def prepare_ml_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Prepare data for ML training"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'target_name': target_column
        }