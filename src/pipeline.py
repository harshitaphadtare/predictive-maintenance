"""
Fixed ML Pipeline for Predictive Maintenance
This version ensures consistent feature engineering between training and test data.
"""

import numpy as np
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import local modules
from config import *
from data_prep import generate_labels
from features import (
    add_rolling_stats, 
    add_lag_features, 
    scale_features, 
    add_delta_features, 
    drop_correlated_features
)
from models import model_configs, train_model, save_model
from evaluate import regression_metrics, classification_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictiveMaintenancePipeline:
    """
    ML pipeline for predictive maintenance with consistent feature engineering
    """
    
    def __init__(self, dataset_id=DATASET_ID, failure_threshold=FAILURE_THRESHOLD):
        self.dataset_id = dataset_id
        self.failure_threshold = failure_threshold
        self.train_data = None
        self.test_data = None
        self.feature_columns = None
        self.models = {}
        self.results = {}
        
        # Store feature engineering parameters for consistency
        self.flat_sensors_to_drop = None
        self.correlated_features_to_drop = None
        self.scaler = None
        
    def load_and_prepare_data(self):
        """Load and prepare training and test data"""
        logger.info("\n\nLoading and preparing data...")
        
        # Load and generate labels for training data
        self.train_data = generate_labels(
            dataset_id=self.dataset_id, 
            dataset_type="train", 
            failure_threshold=self.failure_threshold
        )
        
        # Load and generate labels for test data
        self.test_data = generate_labels(
            dataset_id=self.dataset_id, 
            dataset_type="test", 
            failure_threshold=self.failure_threshold
        )
        
        logger.info(f"Training data shape: {self.train_data.shape}")
        logger.info(f"Test data shape: {self.test_data.shape}")
        
        return self
    
    def engineer_features(self, drop_correlated=True):
        """Apply feature engineering with consistency between train and test data"""
        logger.info("\n\nEngineering features...")
        
        # Get configuration parameters
        rolling_window = FEATURE_ENGINEERING_CONFIG['rolling_window']
        lag_features = FEATURE_ENGINEERING_CONFIG['lag_features']
        flat_sensor_threshold = FEATURE_ENGINEERING_CONFIG['flat_sensor_threshold']
        correlation_threshold = FEATURE_ENGINEERING_CONFIG['correlation_threshold']
        
        # Step 1: Identify flat sensors from training data only
        sensor_cols = [col for col in self.train_data.columns if 'sensor_' in col]
        self.flat_sensors_to_drop = [
            col for col in sensor_cols 
            if self.train_data[col].std() < flat_sensor_threshold
        ]
        
        logger.info(f"Dropping {len(self.flat_sensors_to_drop)} flat sensors: {self.flat_sensors_to_drop}")
        
        # Step 2: Apply consistent feature engineering to both datasets
        self.train_data = self._apply_feature_engineering_consistent(
            self.train_data, 
            rolling_window=rolling_window,
            lag_features=lag_features,
            correlation_threshold=correlation_threshold,
            drop_correlated=drop_correlated,
            is_training=True
        )
        
        self.test_data = self._apply_feature_engineering_consistent(
            self.test_data, 
            rolling_window=rolling_window,
            lag_features=lag_features,
            correlation_threshold=correlation_threshold,
            drop_correlated=drop_correlated,
            is_training=False
        )
        
        # Step 3: Ensure both datasets have the same columns
        self._align_columns()
        
        # Step 4: Identify feature columns
        exclude_cols = ['unit_number', 'time_in_cycles', 'RUL', 'label']
        self.feature_columns = [col for col in self.train_data.columns 
                              if col not in exclude_cols]
        
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Training data shape after feature engineering: {self.train_data.shape}")
        logger.info(f"Test data shape after feature engineering: {self.test_data.shape}")
        
        return self
    
    def _apply_feature_engineering_consistent(self, df, rolling_window=3, 
                                           lag_features=[1,2], correlation_threshold=0.95,
                                           drop_correlated=True, is_training=True):
        """Apply feature engineering with consistency between train and test data"""
        
        # Step 1: Drop flat sensors (use the same list for both train and test)
        if self.flat_sensors_to_drop:
            df = df.drop(columns=self.flat_sensors_to_drop)
        
        # Step 2: Identify remaining sensor columns
        sensor_cols = [col for col in df.columns if 'sensor_' in col]
        
        # Step 3: Add delta features
        df = add_delta_features(df, sensor_cols)
        
        # Step 4: Add lag features
        df = add_lag_features(df, sensor_cols, lags=lag_features)
        
        # Step 5: Add rolling statistics
        df = add_rolling_stats(df, sensor_cols, window=rolling_window)
        
        # Step 6: Handle missing values from lag and rolling
        df = df.dropna()
        
        # Step 7: Normalize features (fit scaler only on training data)
        feature_cols = [col for col in df.columns if ('sensor_' in col) or 
                       ('roll' in col) or ('lag' in col) or ('delta' in col)]
        
        # Convert feature columns to float before scaling to avoid dtype warnings
        for col in feature_cols:
            df[col] = df[col].astype('float64')
            
        if is_training:
            # Fit scaler on training data
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            df.loc[:, feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            # Transform test data using fitted scaler
            if self.scaler is not None:
                df.loc[:, feature_cols] = self.scaler.transform(df[feature_cols])
        
        # Step 8: Drop correlated features (only on training data, then apply to test)
        if drop_correlated and is_training:
            df, self.correlated_features_to_drop = drop_correlated_features(
                df, df.columns.to_list(), threshold=correlation_threshold
            )
            logger.info(f"Dropped {len(self.correlated_features_to_drop)} highly correlated features")
        elif not is_training and self.correlated_features_to_drop:
            # Apply the same feature dropping to test data
            df = df.drop(columns=self.correlated_features_to_drop)
        
        return df
    
    def _align_columns(self):
        """Ensure both train and test data have the same columns"""
        train_cols = set(self.train_data.columns)
        test_cols = set(self.test_data.columns)
        
        # Find missing columns in each dataset
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        
        if missing_in_test:
            logger.warning(f"Adding missing columns to test data: {missing_in_test}")
            for col in missing_in_test:
                self.test_data[col] = 0  # Fill with zeros or appropriate default
        
        if missing_in_train:
            logger.warning(f"Adding missing columns to train data: {missing_in_train}")
            for col in missing_in_train:
                self.train_data[col] = 0  # Fill with zeros or appropriate default
        
        # Ensure same column order
        self.test_data = self.test_data[self.train_data.columns]
    
    def prepare_training_data(self, task='regression'):
        """Prepare X and y for training"""
        if task == 'regression':
            y_col = 'RUL'
        else:  # classification
            y_col = 'label'
            
        X = self.train_data[self.feature_columns]
        y = self.train_data[y_col]
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state']
        )
        
        return X_train, X_val, y_train, y_val, X, y
    
    def train_models(self, task='regression'):
        """Train all models for the specified task"""
        logger.info(f"\n\nTraining {task} models...")
        
        X_train, X_val, y_train, y_val, X, y = self.prepare_training_data(task)
        
        task_models = MODEL_CONFIGS[task]
        
        for model_name in task_models:
            logger.info(f"Training {model_name}...")
            
            try:
                # Get model configuration
                model_config = model_configs[model_name]
                model = model_config['model']
                param_grid = model_config['params']
                
                # Train model
                best_model, best_params, best_score = train_model(
                    X_train, y_train, model, param_grid, task
                )
                
                # Evaluate on validation set
                y_val_pred = best_model.predict(X_val)
                
                if task == 'regression':
                    val_metrics = regression_metrics(y_val, y_val_pred)
                else:
                    val_metrics = classification_metrics(y_val, y_val_pred)
                    val_metrics['Accuracy'] = accuracy_score(y_val, y_val_pred)
                
                # Store results
                self.models[f"{task}_{model_name}"] = {
                    'model': best_model,
                    'params': best_params,
                    'cv_score': best_score,
                    'val_metrics': val_metrics
                }
                
                logger.info(f"{model_name} - Best CV Score: {best_score:.4f}")
                logger.info(f"{model_name} - Validation Metrics: {val_metrics}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return self
    
    def identify_best_model(self, task='regression'):
        """Identify the best model for the given task based on test performance"""
        if f"{task}_test_results" not in self.results:
            logger.warning(f"No test results found for {task}")
            return None
        
        test_results = self.results[f"{task}_test_results"]
        
        if task == 'regression':
            # For regression, use R2 score (higher is better)
            best_model = None
            best_score = -float('inf')
            
            for model_name, metrics in test_results.items():
                r2_score = metrics.get('R2:', -float('inf'))
                if r2_score > best_score:
                    best_score = r2_score
                    best_model = model_name
            
            logger.info(f"Best regression model: {best_model} with R2: {best_score:.4f}")
            
        else:  # classification
            # For classification, use F1 score (higher is better)
            best_model = None
            best_score = -float('inf')
            
            for model_name, metrics in test_results.items():
                f1_score = metrics.get('F1:', -float('inf'))
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = model_name
            
            logger.info(f"Best classification model: {best_model} with F1: {best_score:.4f}")
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'all_models': test_results
        }
    
    def evaluate_on_test_set(self, task='regression'):
        """Evaluate all trained models on the test set and identify the best model"""
        logger.info(f"\n\nEvaluating {task} models on test set...")
        
        # Ensure test data has the required columns
        missing_cols = set(self.feature_columns) - set(self.test_data.columns)
        if missing_cols:
            logger.error(f"Missing columns in test data: {missing_cols}")
            raise ValueError(f"Test data missing required columns: {missing_cols}")
        
        X_test = self.test_data[self.feature_columns]
        
        if task == 'regression':
            y_test = self.test_data['RUL']
        else:
            y_test = self.test_data['label']
        
        test_results = {}
        
        for model_key, model_info in self.models.items():
            if task in model_key:
                model = model_info['model']
                y_test_pred = model.predict(X_test)
                
                if task == 'regression':
                    test_metrics = regression_metrics(y_test, y_test_pred)
                else:
                    test_metrics = classification_metrics(y_test, y_test_pred)
                    test_metrics['Accuracy'] = accuracy_score(y_test, y_test_pred)
                
                test_results[model_key] = test_metrics
                logger.info(f"{model_key} - Test Metrics: {test_metrics}")
        
        self.results[f"{task}_test_results"] = test_results
        
        # Identify and store the best model
        best_model_info = self.identify_best_model(task)
        if best_model_info:
            self.results[f"{task}_best_model"] = best_model_info
        
        return test_results
    
    def get_best_model_info(self, task='regression'):
        """Get information about the best model for a given task"""
        best_model_key = f"{task}_best_model"
        if best_model_key in self.results:
            return self.results[best_model_key]
        return None
    
    def save_models(self):
        """Save all trained models and best models separately"""
        logger.info("\n\nSaving models...")
        
        # Save all models
        for model_name, model_info in self.models.items():
            try:
                save_model(model_info['model'], model_name)
                logger.info(f"Saved model: {model_name}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save best models separately
        for task in ['regression', 'classification']:
            best_model_info = self.get_best_model_info(task)
            if best_model_info and best_model_info['best_model']:
                best_model_name = best_model_info['best_model']
                if best_model_name in self.models:
                    try:
                        # Save best model with special naming
                        best_model = self.models[best_model_name]['model']
                        save_model(best_model, f"best_{task}_model")
                        logger.info(f"Saved best {task} model: {best_model_name}")
                    except Exception as e:
                        logger.error(f"Error saving best {task} model: {str(e)}")
        
        return self
    
    def save_results(self):
        """Save pipeline results and metrics"""
        logger.info("\n\nSaving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy arrays to lists for JSON serialization
        results_for_save = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                if 'all_models' in value:
                    # Handle best model results with nested all_models
                    results_for_save[key] = {
                        'best_model': value['best_model'],
                        'best_score': float(value['best_score']) if isinstance(value['best_score'], (np.integer, np.floating)) else value['best_score'],
                        'all_models': {
                            k: {metric: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                for metric, v in model_metrics.items()}
                            for k, model_metrics in value['all_models'].items()
                        }
                    }
                else:
                    # Handle regular test results
                    results_for_save[key] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in value.items()
                    }
            else:
                results_for_save[key] = value
        
        # Save results to JSON
        results_file = os.path.join(REPORTS_PATH, f"pipeline_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results_for_save, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return self
    
    def run_pipeline(self, tasks=['regression', 'classification']):
        """Run the complete pipeline"""
        logger.info("\n\nStarting ML Pipeline...")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Engineer features
            self.engineer_features(drop_correlated=FEATURE_ENGINEERING_CONFIG['drop_correlated'])
            
            # Step 3: Train models for each task
            for task in tasks:
                self.train_models(task)
                self.evaluate_on_test_set(task)
            
            # Step 4: Save models and results
            self.save_models()
            self.save_results()
            
            logger.info("\n\nPipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"\n\nPipeline failed: {str(e)}")
            raise
        
        return self


def main():
    """Main function to run the fixed pipeline"""
    # Initialize pipeline
    pipeline = PredictiveMaintenancePipeline()
    
    # Run pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 