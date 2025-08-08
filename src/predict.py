"""
Prediction module for the Predictive Maintenance system
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Union, Dict, List, Optional

from config import *
from data_prep import load_data
from features import (
    add_rolling_stats, 
    add_lag_features, 
    add_delta_features, 
    drop_correlated_features
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictiveMaintenancePredictor:
    """
    Predictor class for making predictions using trained models
    """
    
    def __init__(self, dataset_id=DATASET_ID, failure_threshold=FAILURE_THRESHOLD):
        self.dataset_id = dataset_id
        self.failure_threshold = failure_threshold
        self.regression_model = None
        self.classification_model = None
        self.feature_columns = None
        self.scaler = None
        self.flat_sensors_to_drop = None
        self.correlated_features_to_drop = None
        
        # Load best models and configuration
        self._load_models()
        self._load_configuration()
    
    def _load_models(self):
        """Load the best trained models"""
        logger.info("Loading best models...")
        
        # Load regression model
        regression_model_path = os.path.join(MODELS_PATH, "best_regression_model.joblib")
        if os.path.exists(regression_model_path):
            self.regression_model = joblib.load(regression_model_path)
            logger.info("Loaded best regression model")
        else:
            logger.warning("Best regression model not found")
        
        # Load classification model
        classification_model_path = os.path.join(MODELS_PATH, "best_classification_model.joblib")
        if os.path.exists(classification_model_path):
            self.classification_model = joblib.load(classification_model_path)
            logger.info("Loaded best classification model")
        else:
            logger.warning("Best classification model not found")
    
    def _load_configuration(self):
        """Load feature engineering configuration from training data"""
        logger.info("Loading feature engineering configuration...")
        
        # Load training data to get feature engineering parameters
        train_data = load_data(dataset_id=self.dataset_id, dataset_type="train")
        
        # Identify flat sensors from training data
        sensor_cols = [col for col in train_data.columns if 'sensor_' in col]
        flat_sensor_threshold = FEATURE_ENGINEERING_CONFIG['flat_sensor_threshold']
        self.flat_sensors_to_drop = [
            col for col in sensor_cols 
            if train_data[col].std() < flat_sensor_threshold
        ]
        
        # Apply feature engineering to training data to get feature columns
        train_data = self._apply_feature_engineering(train_data, is_training=True)
        
        # Get feature columns (exclude target columns)
        exclude_cols = ['unit_number', 'time_in_cycles', 'RUL', 'label']
        self.feature_columns = [col for col in train_data.columns 
                              if col not in exclude_cols]
        
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Flat sensors to drop: {self.flat_sensors_to_drop}")
    
    def _apply_feature_engineering(self, df, is_training=True):
        """Apply feature engineering to input data"""
        
        # Get configuration parameters
        rolling_window = FEATURE_ENGINEERING_CONFIG['rolling_window']
        lag_features = FEATURE_ENGINEERING_CONFIG['lag_features']
        correlation_threshold = FEATURE_ENGINEERING_CONFIG['correlation_threshold']
        
        # Step 1: Drop flat sensors
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
        
        # Step 7: Normalize features
        feature_cols = [col for col in df.columns if ('sensor_' in col) or 
                       ('roll' in col) or ('lag' in col) or ('delta' in col)]
        
        # Convert feature columns to float before scaling
        for col in feature_cols:
            df[col] = df[col].astype('float64')
        
        if is_training:
            # Fit scaler on training data
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            df.loc[:, feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            # Transform using fitted scaler
            if self.scaler is not None:
                df.loc[:, feature_cols] = self.scaler.transform(df[feature_cols])
        
        # Step 8: Drop correlated features (only on training data)
        if is_training:
            df, self.correlated_features_to_drop = drop_correlated_features(
                df, df.columns.to_list(), threshold=correlation_threshold
            )
        elif self.correlated_features_to_drop:
            # Apply the same feature dropping to test data
            df = df.drop(columns=self.correlated_features_to_drop)
        
        return df
    

    
    def predict_from_file(self, file_path: str) -> List[Dict]:
        """
        Make predictions for data from a file
        
        Args:
            file_path: Path to the input file (CSV or text file)
            
        Returns:
            List of dictionaries with predictions
        """
        logger.info(f"Making predictions from file: {file_path}")
        
        # Load data from file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # Assume it's a text file with space-separated values
            col_names = ['unit_number','time_in_cycles'] + \
                       [f'op_setting_{i}' for i in range(1,4)] + \
                       [f'sensor_{i}' for i in range(1,22)]
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
            df.dropna(axis=1, how='all', inplace=True)
        
        # Apply feature engineering
        df = self._apply_feature_engineering(df, is_training=False)
        
        # Ensure we have the required features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0
        
        # Select only the required features
        X = df[self.feature_columns]
        
        # Make predictions
        predictions = []
        
        for i in range(len(df)):
            pred = {}
            
            if self.regression_model is not None:
                rul_prediction = self.regression_model.predict(X.iloc[[i]])[0]
                pred['RUL'] = float(rul_prediction)
            
            if self.classification_model is not None:
                failure_prob = self.classification_model.predict_proba(X.iloc[[i]])[0]
                failure_prediction = self.classification_model.predict(X.iloc[[i]])[0]
                pred['failure_probability'] = float(failure_prob[1])
                pred['failure_prediction'] = int(failure_prediction)
                pred['failure_status'] = "FAILURE IMMINENT" if failure_prediction == 1 else "NORMAL"
            
            predictions.append(pred)
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions


def create_sample_file(output_path="sample_input.txt", num_units=3, cycles_per_unit=10):
    """
    Create a sample input file for testing with multiple units and cycles
    
    Args:
        output_path: Path to save the sample file
        num_units: Number of units to include
        cycles_per_unit: Number of cycles per unit
    """
    import random
    
    # Base sensor values
    base_sensors = {
        'sensor_1': 518.67,
        'sensor_2': 641.82,
        'sensor_3': 1589.70,
        'sensor_4': 1400.60,
        'sensor_5': 14.62,
        'sensor_6': 21.61,
        'sensor_7': 554.36,
        'sensor_8': 2388.06,
        'sensor_9': 9046.19,
        'sensor_10': 1.3,
        'sensor_11': 47.47,
        'sensor_12': 521.66,
        'sensor_13': 2388.02,
        'sensor_14': 8138.62,
        'sensor_15': 8.4195,
        'sensor_16': 0.03,
        'sensor_17': 392,
        'sensor_18': 2388,
        'sensor_19': 100,
        'sensor_20': 39.06,
        'sensor_21': 23.4190
    }
    
    # Base operating settings
    base_op_settings = {
        'op_setting_1': 0.4592,
        'op_setting_2': 0.0003,
        'op_setting_3': 100.0
    }
    
    sample_data = []
    
    for unit in range(1, num_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            # Add some variation to the data
            variation_factor = 1 + random.uniform(-0.1, 0.1)  # ±10% variation
            
            row = [unit, cycle]
            
            # Add operating settings with variation
            for key in ['op_setting_1', 'op_setting_2', 'op_setting_3']:
                value = base_op_settings[key] * variation_factor
                row.append(value)
            
            # Add sensor values with variation
            for i in range(1, 22):
                value = base_sensors[f'sensor_{i}'] * variation_factor
                row.append(value)
            
            sample_data.append(row)
    
    # Write to file
    with open(output_path, 'w') as f:
        for row in sample_data:
            f.write(' '.join(map(str, row)) + '\n')
    
    logger.info(f"Sample input file created: {output_path}")
    logger.info(f"Contains {len(sample_data)} records ({num_units} units, {cycles_per_unit} cycles each)")
    return output_path


if __name__ == "__main__":
    # Test the predictor with file-based prediction
    predictor = PredictiveMaintenancePredictor()
    
    # Create sample file
    sample_file = create_sample_file()
    
    # Test with sample file
    predictions = predictor.predict_from_file(sample_file)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predictions for {len(predictions)} samples")
    if predictions:
        rul_values = [p.get('RUL', 0) for p in predictions if 'RUL' in p]
        failure_predictions = [p.get('failure_prediction', 0) for p in predictions if 'failure_prediction' in p]
        
        if rul_values:
            print(f"Average RUL: {sum(rul_values)/len(rul_values):.2f} cycles")
            print(f"Min RUL: {min(rul_values):.2f} cycles")
            print(f"Max RUL: {max(rul_values):.2f} cycles")
        
        if failure_predictions:
            failure_count = sum(failure_predictions)
            print(f"Failure predictions: {failure_count}/{len(failure_predictions)} ({failure_count/len(failure_predictions)*100:.1f}%)")
    print("="*60) 
