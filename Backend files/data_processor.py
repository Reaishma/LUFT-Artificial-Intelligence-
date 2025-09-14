import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Any, Tuple, Optional
import os

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def load_data(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_type.lower() in ['xlsx', 'xls', 'excel']:
                df = pd.read_excel(file_path)
            elif file_type.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset and return metadata"""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'statistics': {}
        }
        
        # Basic statistics for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            analysis['statistics'] = numeric_df.describe().to_dict()
        
        # Unique values count for categorical columns
        analysis['unique_counts'] = {}
        for col in analysis['categorical_columns']:
            analysis['unique_counts'][col] = df[col].nunique()
            
        return analysis

    def preprocess_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess data based on configuration"""
        preprocessed_df = df.copy()
        preprocessing_info = {
            'steps_applied': [],
            'transformers': {}
        }
        
        # Handle missing values
        if config.get('handle_missing', True):
            missing_strategy = config.get('missing_strategy', 'mean')
            preprocessed_df, imputer_info = self._handle_missing_values(
                preprocessed_df, missing_strategy
            )
            preprocessing_info['steps_applied'].append('missing_values_handled')
            preprocessing_info['transformers']['imputers'] = imputer_info

        # Encode categorical variables
        if config.get('encode_categorical', True):
            preprocessed_df, encoder_info = self._encode_categorical(preprocessed_df)
            preprocessing_info['steps_applied'].append('categorical_encoded')
            preprocessing_info['transformers']['encoders'] = encoder_info

        # Scale numerical features
        if config.get('scale_features', True):
            scaling_method = config.get('scaling_method', 'standard')
            preprocessed_df, scaler_info = self._scale_features(
                preprocessed_df, scaling_method
            )
            preprocessing_info['steps_applied'].append('features_scaled')
            preprocessing_info['transformers']['scalers'] = scaler_info

        # Feature engineering
        if config.get('feature_engineering', False):
            preprocessed_df = self._engineer_features(preprocessed_df, config)
            preprocessing_info['steps_applied'].append('features_engineered')

        return preprocessed_df, preprocessing_info

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values in the dataset"""
        imputer_info = {}
        
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            else:
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            imputer_info['numeric'] = {
                'strategy': strategy,
                'columns': list(numeric_cols)
            }

        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
            imputer_info['categorical'] = {
                'strategy': 'most_frequent',
                'columns': list(categorical_cols)
            }

        return df, imputer_info

    def _encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables"""
        encoder_info = {}
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoder_info[col] = {
                'classes': list(le.classes_),
                'encoded_column': col + '_encoded'
            }
            # Drop original categorical column
            df = df.drop(columns=[col])
            
        return df, encoder_info

    def _scale_features(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features"""
        scaler_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()  # default
            
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            scaler_info = {
                'method': method,
                'columns': list(numeric_cols),
                'parameters': {
                    'mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                    'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                }
            }
            
        return df, scaler_info

    def _engineer_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create new features based on existing ones"""
        # Create polynomial features for numeric columns if requested
        if config.get('polynomial_features', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Limit to avoid explosion
            for col in numeric_cols:
                df[f'{col}_squared'] = df[col] ** 2
        
        # Create interaction features if requested
        if config.get('interaction_features', False):
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) >= 2:
                df[f'{numeric_cols[0]}_x_{numeric_cols[1]}'] = df[numeric_cols[0]] * df[numeric_cols[1]]
        
        return df

    def prepare_ml_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Prepare data for machine learning"""
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split into train/test sets
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