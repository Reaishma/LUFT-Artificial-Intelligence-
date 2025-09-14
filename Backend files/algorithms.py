import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import joblib
import os
from typing import Dict, Any, Tuple, List
from datetime import datetime

class MLAlgorithms:
    def __init__(self):
        self.models = {}
        self.model_types = {
            'linear_regression': LinearRegression,
            'logistic_regression': LogisticRegression,
            'decision_tree_classifier': DecisionTreeClassifier,
            'decision_tree_regressor': DecisionTreeRegressor,
            'random_forest_classifier': RandomForestClassifier,
            'random_forest_regressor': RandomForestRegressor,
            'svm_classifier': SVC,
            'svm_regressor': SVR,
            'kmeans': KMeans,
            'hierarchical_clustering': AgglomerativeClustering,
            'pca': PCA,
            'neural_network_classifier': MLPClassifier,
            'neural_network_regressor': MLPRegressor
        }

    def create_model(self, algorithm: str, parameters: Dict[str, Any] = None) -> Any:
        """Create a model instance based on algorithm and parameters"""
        if algorithm not in self.model_types:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        model_class = self.model_types[algorithm]
        
        # Default parameters for each algorithm
        default_params = self._get_default_parameters(algorithm)
        
        # Update with user-provided parameters
        if parameters:
            default_params.update(parameters)
        
        return model_class(**default_params)

    def _get_default_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default parameters for each algorithm"""
        defaults = {
            'linear_regression': {},
            'logistic_regression': {'max_iter': 1000, 'random_state': 42},
            'decision_tree_classifier': {'random_state': 42, 'max_depth': 10},
            'decision_tree_regressor': {'random_state': 42, 'max_depth': 10},
            'random_forest_classifier': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
            'random_forest_regressor': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
            'svm_classifier': {'kernel': 'rbf', 'random_state': 42},
            'svm_regressor': {'kernel': 'rbf'},
            'kmeans': {'n_clusters': 3, 'random_state': 42, 'n_init': 10},
            'hierarchical_clustering': {'n_clusters': 3},
            'pca': {'n_components': 2},
            'neural_network_classifier': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42},
            'neural_network_regressor': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': 42}
        }
        return defaults.get(algorithm, {})

    def train_model(self, algorithm: str, X_train, y_train, parameters: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Train a model and return the trained model and training info"""
        try:
            # Create model
            model = self.create_model(algorithm, parameters)
            
            # Record training start time
            start_time = datetime.now()
            
            # Train the model
            if algorithm in ['kmeans', 'hierarchical_clustering', 'pca']:
                # Unsupervised learning - no target needed
                if algorithm == 'pca':
                    model.fit(X_train)
                else:
                    labels = model.fit_predict(X_train)
                    model.labels_ = labels
            else:
                # Supervised learning
                model.fit(X_train, y_train)
            
            # Record training end time
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            training_info = {
                'algorithm': algorithm,
                'training_time': training_time,
                'parameters': parameters or {},
                'feature_count': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'sample_count': X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train),
                'status': 'completed'
            }
            
            return model, training_info
            
        except Exception as e:
            training_info = {
                'algorithm': algorithm,
                'status': 'failed',
                'error': str(e)
            }
            raise Exception(f"Training failed: {str(e)}")

    def evaluate_model(self, model, X_test, y_test, algorithm: str) -> Dict[str, Any]:
        """Evaluate a trained model and return metrics"""
        try:
            metrics = {}
            
            if algorithm in ['kmeans', 'hierarchical_clustering']:
                # Clustering metrics
                from sklearn.metrics import silhouette_score
                if hasattr(model, 'labels_'):
                    predictions = model.labels_[:len(X_test)]
                else:
                    predictions = model.fit_predict(X_test)
                
                if len(set(predictions)) > 1:
                    metrics['silhouette_score'] = float(silhouette_score(X_test, predictions))
                metrics['n_clusters'] = len(set(predictions))
                
            elif algorithm == 'pca':
                # Dimensionality reduction metrics
                metrics['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
                metrics['cumulative_variance'] = np.cumsum(model.explained_variance_ratio_).tolist()
                metrics['n_components'] = model.n_components_
                
            else:
                # Supervised learning metrics
                predictions = model.predict(X_test)
                
                # Classification metrics
                if algorithm in ['logistic_regression', 'decision_tree_classifier', 
                               'random_forest_classifier', 'svm_classifier', 'neural_network_classifier']:
                    metrics['accuracy'] = float(accuracy_score(y_test, predictions))
                    metrics['precision'] = float(precision_score(y_test, predictions, average='weighted', zero_division=0))
                    metrics['recall'] = float(recall_score(y_test, predictions, average='weighted', zero_division=0))
                    metrics['f1_score'] = float(f1_score(y_test, predictions, average='weighted', zero_division=0))
                
                # Regression metrics
                elif algorithm in ['linear_regression', 'decision_tree_regressor', 
                                 'random_forest_regressor', 'svm_regressor', 'neural_network_regressor']:
                    metrics['r2_score'] = float(r2_score(y_test, predictions))
                    metrics['mse'] = float(mean_squared_error(y_test, predictions))
                    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
            
            return metrics
            
        except Exception as e:
            return {'error': f"Evaluation failed: {str(e)}"}

    def save_model(self, model, model_path: str) -> str:
        """Save trained model to file"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            return model_path
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            return joblib.load(model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def create_deep_learning_model(self, architecture: str, input_shape: Tuple, num_classes: int = None) -> tf.keras.Model:
        """Create deep learning models (CNN, RNN, etc.)"""
        if architecture == 'cnn':
            return self._create_cnn(input_shape, num_classes)
        elif architecture == 'rnn':
            return self._create_rnn(input_shape, num_classes)
        elif architecture == 'neural_network':
            return self._create_neural_network(input_shape, num_classes)
        else:
            raise ValueError(f"Unsupported deep learning architecture: {architecture}")

    def _create_cnn(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """Create a Convolutional Neural Network"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def _create_rnn(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """Create a Recurrent Neural Network"""
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def _create_neural_network(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """Create a standard Neural Network"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def get_available_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available algorithms"""
        return {
            'regression': [
                {'name': 'Linear Regression', 'id': 'linear_regression', 'type': 'regression'},
                {'name': 'Decision Tree Regressor', 'id': 'decision_tree_regressor', 'type': 'regression'},
                {'name': 'Random Forest Regressor', 'id': 'random_forest_regressor', 'type': 'regression'},
                {'name': 'SVM Regressor', 'id': 'svm_regressor', 'type': 'regression'},
                {'name': 'Neural Network Regressor', 'id': 'neural_network_regressor', 'type': 'regression'}
            ],
            'classification': [
                {'name': 'Logistic Regression', 'id': 'logistic_regression', 'type': 'classification'},
                {'name': 'Decision Tree Classifier', 'id': 'decision_tree_classifier', 'type': 'classification'},
                {'name': 'Random Forest Classifier', 'id': 'random_forest_classifier', 'type': 'classification'},
                {'name': 'SVM Classifier', 'id': 'svm_classifier', 'type': 'classification'},
                {'name': 'Neural Network Classifier', 'id': 'neural_network_classifier', 'type': 'classification'}
            ],
            'clustering': [
                {'name': 'K-Means', 'id': 'kmeans', 'type': 'clustering'},
                {'name': 'Hierarchical Clustering', 'id': 'hierarchical_clustering', 'type': 'clustering'}
            ],
            'dimensionality_reduction': [
                {'name': 'PCA', 'id': 'pca', 'type': 'dimensionality_reduction'}
            ],
            'deep_learning': [
                {'name': 'Neural Network', 'id': 'neural_network', 'type': 'deep_learning'},
                {'name': 'CNN', 'id': 'cnn', 'type': 'deep_learning'},
                {'name': 'RNN', 'id': 'rnn', 'type': 'deep_learning'}
            ]
        }