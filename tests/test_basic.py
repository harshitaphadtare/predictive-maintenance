"""
Basic tests for the Predictive Maintenance ML Pipeline
"""

import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from config import DATASET_ID, FAILURE_THRESHOLD, FEATURE_ENGINEERING_CONFIG, MODEL_CONFIGS
        from data_prep import load_data, generate_labels
        from features import (
            add_rolling_stats, 
            add_lag_features, 
            scale_features, 
            add_delta_features, 
            drop_correlated_features
        )
        from models import model_configs, train_model, save_model
        from evaluate import regression_metrics, classification_metrics
        from pipeline import PredictiveMaintenancePipeline
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config():
    """Test that configuration is properly set up"""
    from config import DATASET_ID, FAILURE_THRESHOLD, FEATURE_ENGINEERING_CONFIG, MODEL_CONFIGS
    
    assert DATASET_ID == "FD001"
    assert FAILURE_THRESHOLD == 30
    assert "rolling_window" in FEATURE_ENGINEERING_CONFIG
    assert "regression" in MODEL_CONFIGS
    assert "classification" in MODEL_CONFIGS

def test_pipeline_initialization():
    """Test pipeline initialization"""
    from pipeline import PredictiveMaintenancePipeline
    
    pipeline = PredictiveMaintenancePipeline()
    assert pipeline.dataset_id == "FD001"
    assert pipeline.failure_threshold == 30
    assert pipeline.train_data is None
    assert pipeline.test_data is None

def test_feature_functions():
    """Test that feature engineering functions exist"""
    from features import (
        add_rolling_stats, 
        add_lag_features, 
        scale_features, 
        add_delta_features, 
        drop_correlated_features
    )
    
    # Test that functions are callable
    assert callable(add_rolling_stats)
    assert callable(add_lag_features)
    assert callable(scale_features)
    assert callable(add_delta_features)
    assert callable(drop_correlated_features)

def test_model_configs():
    """Test that model configurations exist"""
    from models import model_configs
    
    expected_models = ["ridge", "lasso", "rf_reg", "polynomial", "logistic_reg", "rf_clf", "svm"]
    
    for model_name in expected_models:
        assert model_name in model_configs
        assert "model" in model_configs[model_name]
        assert "params" in model_configs[model_name]

def test_evaluation_functions():
    """Test that evaluation functions exist"""
    from evaluate import regression_metrics, classification_metrics
    
    assert callable(regression_metrics)
    assert callable(classification_metrics)

if __name__ == "__main__":
    pytest.main([__file__]) 