import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any, Tuple, Optional, Union
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data preprocessing and feature engineering system
    Handles missing values, encoding, scaling, feature selection, and data quality assessment
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.transformers = {}
        self.preprocessing_pipeline = None
        self.data_quality_report = {}
        
    def load_data(self, file_path: str, file_type: str = None) -> pd.DataFrame:
        """Load data from various file formats with automatic format detection"""
        try:
            if not file_type:
                file_type = self._detect_file_type(file_path)
            
            if file_type.lower() == 'csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
                else:
                    # Try different separators
                    for sep in [',', ';', '\t', '|']:
                        try:
                            df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                            if df.shape[1] > 1:  # Likely correct separator
                                break
                        except:
                            continue
                            
            elif file_type.lower() in ['xlsx', 'xls', 'excel']:
                df = pd.read_excel(file_path, engine='openpyxl' if file_type == 'xlsx' else None)
                
            elif file_type.lower() == 'json':
                # Handle different JSON structures
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.DataFrame([data])
                else:
                    df = pd.read_json(file_path)
                    
            elif file_type.lower() == 'parquet':
                df = pd.read_parquet(file_path)
                
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Basic data validation
            if df.empty:
                raise ValueError("Dataset is empty")
                
            if df.shape[1] == 0:
                raise ValueError("Dataset has no columns")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _detect_file_type(self, file_path: str) -> str:
        """Automatically detect file type from extension"""
        _, ext = os.path.splitext(file_path.lower())
        ext_map = {
            '.csv': 'csv',
            '.xlsx': 'xlsx',
            '.xls': 'xls',
            '.json': 'json',
            '.parquet': 'parquet'
        }
        return ext_map.get(ext, 'csv')

    def analyze_data(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Comprehensive data analysis and quality assessment"""
        analysis = {
            'basic_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'total_size': df.size
            },
            'data_types': {
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
                'boolean_columns': list(df.select_dtypes(include=['bool']).columns)
            },
            'missing_values': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'data_quality': self._assess_data_quality(df),
            'statistics': {},
            'recommendations': []
        }
        
        # Numeric column statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            analysis['statistics']['numeric'] = {
                'descriptive': numeric_df.describe().to_dict(),
                'correlation_matrix': numeric_df.corr().to_dict(),
                'skewness': numeric_df.skew().to_dict(),
                'outliers': self._detect_outliers(numeric_df)
            }
        
        # Categorical column analysis
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            analysis['statistics']['categorical'] = {}
            for col in categorical_df.columns:
                analysis['statistics']['categorical'][col] = {
                    'unique_count': df[col].nunique(),
                    'unique_values': df[col].unique()[:20].tolist(),  # First 20 unique values
                    'value_counts': df[col].value_counts().head(10).to_dict(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                }
        
        # Target variable analysis
        if target_column and target_column in df.columns:
            analysis['target_analysis'] = self._analyze_target_variable(df, target_column)
            analysis['problem_type'] = self._infer_problem_type(df[target_column])
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        self.data_quality_report = analysis
        return analysis

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality_issues = []
        quality_score = 100
        
        # Missing values
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 50:
            quality_issues.append("High percentage of missing values (>50%)")
            quality_score -= 30
        elif missing_percentage > 20:
            quality_issues.append("Moderate missing values (>20%)")
            quality_score -= 15
            
        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"{duplicate_count} duplicate rows found")
            quality_score -= min(20, duplicate_count / len(df) * 100)
            
        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_issues.append(f"Constant columns found: {constant_cols}")
            quality_score -= len(constant_cols) * 5
            
        # High cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.5]
        if high_cardinality_cols:
            quality_issues.append(f"High cardinality categorical columns: {high_cardinality_cols}")
            quality_score -= len(high_cardinality_cols) * 10
        
        return {
            'quality_score': max(0, quality_score),
            'quality_issues': quality_issues,
            'duplicate_rows': duplicate_count,
            'constant_columns': constant_cols,
            'high_cardinality_columns': high_cardinality_cols
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        return outliers

    def _analyze_target_variable(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable characteristics"""
        target_series = df[target_column]
        
        analysis = {
            'type': str(target_series.dtype),
            'unique_count': target_series.nunique(),
            'missing_count': target_series.isnull().sum(),
            'missing_percentage': (target_series.isnull().sum() / len(target_series)) * 100
        }
        
        if target_series.dtype in ['object', 'category']:
            # Categorical target
            analysis['class_distribution'] = target_series.value_counts().to_dict()
            analysis['class_balance'] = self._assess_class_balance(target_series.value_counts())
        else:
            # Numeric target
            analysis['statistics'] = {
                'mean': float(target_series.mean()),
                'median': float(target_series.median()),
                'std': float(target_series.std()),
                'min': float(target_series.min()),
                'max': float(target_series.max()),
                'skewness': float(target_series.skew())
            }
        
        return analysis

    def _assess_class_balance(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Assess class balance for classification problems"""
        total = value_counts.sum()
        proportions = value_counts / total
        
        # Calculate imbalance ratio
        max_prop = proportions.max()
        min_prop = proportions.min()
        imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        balance_status = "balanced"
        if imbalance_ratio > 10:
            balance_status = "highly_imbalanced"
        elif imbalance_ratio > 3:
            balance_status = "moderately_imbalanced"
        elif imbalance_ratio > 1.5:
            balance_status = "slightly_imbalanced"
        
        return {
            'status': balance_status,
            'imbalance_ratio': float(imbalance_ratio),
            'class_proportions': proportions.to_dict(),
            'minority_class_size': int(value_counts.min()),
            'majority_class_size': int(value_counts.max())
        }

    def _infer_problem_type(self, target_series: pd.Series) -> str:
        """Infer ML problem type from target variable"""
        if target_series.dtype in ['object', 'category', 'bool']:
            unique_count = target_series.nunique()
            if unique_count == 2:
                return 'binary_classification'
            elif unique_count <= 20:
                return 'multiclass_classification'
            else:
                return 'high_cardinality_classification'
        else:
            if target_series.nunique() / len(target_series) > 0.9:
                return 'regression'
            else:
                return 'discrete_regression'

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate data preprocessing recommendations"""
        recommendations = []
        
        # Missing values recommendations
        missing_percentage = analysis['missing_values']['total_missing'] / analysis['basic_info']['total_size'] * 100
        if missing_percentage > 5:
            recommendations.append("Consider handling missing values with imputation strategies")
        
        # Scaling recommendations
        if analysis['data_types']['numeric_columns']:
            recommendations.append("Scale numeric features for algorithms sensitive to feature scales")
        
        # Encoding recommendations
        if analysis['data_types']['categorical_columns']:
            high_cardinality = [col for col in analysis['data_types']['categorical_columns'] 
                              if col in analysis.get('statistics', {}).get('categorical', {}) 
                              and analysis['statistics']['categorical'][col]['unique_count'] > 20]
            if high_cardinality:
                recommendations.append(f"Use target encoding for high-cardinality categorical features: {high_cardinality}")
            else:
                recommendations.append("Use one-hot encoding for categorical features")
        
        # Outlier recommendations
        if 'outliers' in analysis.get('statistics', {}).get('numeric', {}):
            outlier_cols = [col for col, count in analysis['statistics']['numeric']['outliers'].items() if count > 0]
            if outlier_cols:
                recommendations.append(f"Consider outlier treatment for columns: {outlier_cols}")
        
        # Feature selection recommendations
        if len(analysis['basic_info']['columns']) > 100:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        return recommendations

    def preprocess_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive data preprocessing pipeline"""
        preprocessed_df = df.copy()
        preprocessing_info = {
            'steps_applied': [],
            'transformers': {},
            'preprocessing_config': config,
            'original_shape': df.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Handle missing values
        if config.get('handle_missing', True):
            preprocessed_df, missing_info = self._handle_missing_values(
                preprocessed_df, 
                config.get('missing_strategy', 'auto'),
                config.get('missing_threshold', 0.5)
            )
            preprocessing_info['steps_applied'].append('missing_values_handled')
            preprocessing_info['transformers']['missing'] = missing_info

        # Step 2: Remove duplicates
        if config.get('remove_duplicates', True):
            initial_rows = len(preprocessed_df)
            preprocessed_df = preprocessed_df.drop_duplicates()
            duplicates_removed = initial_rows - len(preprocessed_df)
            if duplicates_removed > 0:
                preprocessing_info['steps_applied'].append('duplicates_removed')
                preprocessing_info['transformers']['duplicates'] = {'removed_count': duplicates_removed}

        # Step 3: Handle outliers
        if config.get('handle_outliers', False):
            preprocessed_df, outlier_info = self._handle_outliers(
                preprocessed_df,
                config.get('outlier_method', 'iqr'),
                config.get('outlier_threshold', 1.5)
            )
            preprocessing_info['steps_applied'].append('outliers_handled')
            preprocessing_info['transformers']['outliers'] = outlier_info

        # Step 4: Feature engineering
        if config.get('feature_engineering', False):
            preprocessed_df, feature_info = self._engineer_features(
                preprocessed_df, 
                config.get('feature_engineering_config', {})
            )
            preprocessing_info['steps_applied'].append('features_engineered')
            preprocessing_info['transformers']['feature_engineering'] = feature_info

        # Step 5: Encode categorical variables
        if config.get('encode_categorical', True):
            preprocessed_df, encoding_info = self._encode_categorical_variables(
                preprocessed_df,
                config.get('encoding_method', 'auto'),
                config.get('target_column')
            )
            preprocessing_info['steps_applied'].append('categorical_encoded')
            preprocessing_info['transformers']['encoding'] = encoding_info

        # Step 6: Scale numerical features
        if config.get('scale_features', True):
            preprocessed_df, scaling_info = self._scale_numerical_features(
                preprocessed_df,
                config.get('scaling_method', 'standard')
            )
            preprocessing_info['steps_applied'].append('features_scaled')
            preprocessing_info['transformers']['scaling'] = scaling_info

        # Step 7: Feature selection
        if config.get('feature_selection', False):
            preprocessed_df, selection_info = self._select_features(
                preprocessed_df,
                config.get('target_column'),
                config.get('feature_selection_method', 'auto'),
                config.get('max_features', None)
            )
            preprocessing_info['steps_applied'].append('features_selected')
            preprocessing_info['transformers']['feature_selection'] = selection_info

        # Final preprocessing info
        preprocessing_info['final_shape'] = preprocessed_df.shape
        preprocessing_info['shape_change'] = {
            'rows_change': preprocessed_df.shape[0] - df.shape[0],
            'columns_change': preprocessed_df.shape[1] - df.shape[1]
        }
        
        return preprocessed_df, preprocessing_info

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str, threshold: float) -> Tuple[pd.DataFrame, Dict]:
        """Advanced missing value handling"""
        missing_info = {'strategy': strategy, 'columns_processed': [], 'imputation_values': {}}
        
        # Remove columns with too many missing values
        missing_percentage = df.isnull().sum() / len(df)
        columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            missing_info['dropped_columns'] = columns_to_drop
        
        # Handle remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'auto':
                # Choose strategy based on missing percentage
                for col in numeric_cols:
                    col_missing_pct = df[col].isnull().sum() / len(df)
                    if col_missing_pct < 0.1:
                        imputer = SimpleImputer(strategy='mean')
                    elif col_missing_pct < 0.3:
                        imputer = SimpleImputer(strategy='median')
                    else:
                        imputer = KNNImputer(n_neighbors=5)
                    
                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                    missing_info['columns_processed'].append(col)
            else:
                imputer_strategy = 'mean' if strategy == 'mean' else 'median'
                imputer = SimpleImputer(strategy=imputer_strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                missing_info['columns_processed'].extend(numeric_cols.tolist())

        # Categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                most_frequent = df[col].mode()
                if len(most_frequent) > 0:
                    df[col] = df[col].fillna(most_frequent.iloc[0])
                    missing_info['imputation_values'][col] = most_frequent.iloc[0]
                    missing_info['columns_processed'].append(col)
                else:
                    df[col] = df[col].fillna('Unknown')
                    missing_info['imputation_values'][col] = 'Unknown'
        
        return df, missing_info

    def _handle_outliers(self, df: pd.DataFrame, method: str, threshold: float) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers in numeric columns"""
        outlier_info = {'method': method, 'threshold': threshold, 'outliers_removed': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            initial_count = len(df)
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores <= threshold]
            
            outliers_removed = initial_count - len(df)
            if outliers_removed > 0:
                outlier_info['outliers_removed'][col] = outliers_removed
        
        return df, outlier_info

    def _engineer_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Advanced feature engineering"""
        feature_info = {'new_features': [], 'feature_types': {}}
        original_columns = set(df.columns)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Polynomial features
        if config.get('polynomial_features', False):
            degree = config.get('polynomial_degree', 2)
            selected_cols = numeric_cols[:min(5, len(numeric_cols))]  # Limit to prevent explosion
            
            for col in selected_cols:
                for deg in range(2, degree + 1):
                    new_col = f"{col}_poly_{deg}"
                    df[new_col] = df[col] ** deg
                    feature_info['new_features'].append(new_col)
                    feature_info['feature_types'][new_col] = 'polynomial'

        # Interaction features
        if config.get('interaction_features', False):
            max_interactions = config.get('max_interactions', 10)
            interaction_count = 0
            
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if interaction_count >= max_interactions:
                        break
                    new_col = f"{col1}_x_{col2}"
                    df[new_col] = df[col1] * df[col2]
                    feature_info['new_features'].append(new_col)
                    feature_info['feature_types'][new_col] = 'interaction'
                    interaction_count += 1
                if interaction_count >= max_interactions:
                    break

        # Log transformation
        if config.get('log_transform', False):
            for col in numeric_cols:
                if df[col].min() > 0:  # Only for positive values
                    new_col = f"{col}_log"
                    df[new_col] = np.log1p(df[col])
                    feature_info['new_features'].append(new_col)
                    feature_info['feature_types'][new_col] = 'log_transform'

        # Binning/Discretization
        if config.get('binning', False):
            n_bins = config.get('n_bins', 5)
            for col in numeric_cols:
                new_col = f"{col}_binned"
                df[new_col] = pd.cut(df[col], bins=n_bins, labels=False)
                feature_info['new_features'].append(new_col)
                feature_info['feature_types'][new_col] = 'binned'
        
        return df, feature_info

    def _encode_categorical_variables(self, df: pd.DataFrame, method: str, target_column: str = None) -> Tuple[pd.DataFrame, Dict]:
        """Advanced categorical encoding"""
        encoding_info = {'method': method, 'encoded_columns': {}, 'encoders': {}}
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(target_column)
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if method == 'auto':
                # Choose encoding method based on cardinality
                if unique_count <= 10:
                    chosen_method = 'onehot'
                elif unique_count <= 50:
                    chosen_method = 'target' if target_column else 'label'
                else:
                    chosen_method = 'target' if target_column else 'label'
            else:
                chosen_method = method
            
            if chosen_method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                encoding_info['encoded_columns'][col] = {
                    'method': 'onehot',
                    'new_columns': dummies.columns.tolist(),
                    'original_unique_count': unique_count
                }
                
            elif chosen_method == 'label':
                # Label encoding
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                encoding_info['encoded_columns'][col] = {
                    'method': 'label',
                    'new_column': col + '_encoded',
                    'classes': le.classes_.tolist()
                }
                encoding_info['encoders'][col] = le
                df = df.drop(columns=[col])
                
            elif chosen_method == 'target' and target_column:
                # Target encoding (mean encoding)
                target_means = df.groupby(col)[target_column].mean()
                df[col + '_target_encoded'] = df[col].map(target_means)
                encoding_info['encoded_columns'][col] = {
                    'method': 'target',
                    'new_column': col + '_target_encoded',
                    'target_means': target_means.to_dict()
                }
                df = df.drop(columns=[col])
        
        return df, encoding_info

    def _scale_numerical_features(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features"""
        scaling_info = {'method': method, 'scaled_columns': [], 'scaler_params': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()  # default
            
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            scaling_info['scaled_columns'] = numeric_cols.tolist()
            scaling_info['scaler_params'] = {
                'mean_': getattr(scaler, 'mean_', None),
                'scale_': getattr(scaler, 'scale_', None),
                'center_': getattr(scaler, 'center_', None)
            }
            
            self.scalers[method] = scaler
        
        return df, scaling_info

    def _select_features(self, df: pd.DataFrame, target_column: str, method: str, max_features: int = None) -> Tuple[pd.DataFrame, Dict]:
        """Feature selection"""
        selection_info = {'method': method, 'original_features': len(df.columns), 'selected_features': []}
        
        if not target_column or target_column not in df.columns:
            return df, selection_info
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Determine if classification or regression
        if y.dtype in ['object', 'category'] or y.nunique() <= 20:
            score_func = chi2 if method == 'chi2' else f_classif
        else:
            score_func = f_regression
        
        # Determine number of features to select
        if max_features is None:
            max_features = min(50, len(X.columns) // 2)  # Select half the features, max 50
        
        # Feature selection
        selector = SelectKBest(score_func=score_func, k=min(max_features, len(X.columns)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Create new dataframe with selected features + target
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        df_selected[target_column] = y
        
        selection_info['selected_features'] = selected_features
        selection_info['final_features'] = len(selected_features)
        selection_info['feature_scores'] = dict(zip(selected_features, selector.scores_[selected_mask]))
        
        return df_selected, selection_info

    def prepare_ml_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Prepare data for machine learning with comprehensive validation"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Validate data
        if X.empty or y.empty:
            raise ValueError("Features or target data is empty")
        
        if y.isnull().all():
            raise ValueError("Target variable contains only missing values")
        
        # Remove rows where target is null
        valid_indices = y.notnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 and len(y.unique()) < len(y) * 0.5 else None
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'target_name': target_column,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(X.columns),
            'data_types': X.dtypes.to_dict()
        }

    def export_preprocessing_pipeline(self, file_path: str):
        """Export preprocessing pipeline for reuse"""
        pipeline_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'transformers': self.transformers,
            'data_quality_report': self.data_quality_report,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(pipeline_data, f, indent=2, default=str)

    def get_data_summary(self, df: pd.DataFrame) -> str:
        """Get a human-readable summary of the dataset"""
        summary = []
        summary.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        summary.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        summary.append(f"Column Types: {numeric_count} numeric, {categorical_count} categorical")
        
        # Missing values
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / df.size) * 100
        summary.append(f"Missing Values: {total_missing} ({missing_pct:.1f}%)")
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        summary.append(f"Duplicate Rows: {duplicate_count}")
        
        return "\n".join(summary)