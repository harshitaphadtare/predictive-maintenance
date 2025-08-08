import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
BASE_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports")

# Create directories if they don't exist
os.makedirs(BASE_PROCESSED_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# Dataset configuration
DATASET_ID = "FD001"
FAILURE_THRESHOLD = 30

# Feature engineering parameters
FEATURE_ENGINEERING_CONFIG = {
    "rolling_window": 3,
    "lag_features": [1, 2],
    "drop_correlated": True,
    "correlation_threshold": 0.95,
    "flat_sensor_threshold": 0.1
}

# Model configuration
MODEL_CONFIGS = {
    "regression": ["ridge", "lasso", "rf_reg", "polynomial"],
    "classification": ["logistic_reg", "rf_clf", "svm"]
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Evaluation metrics
REGRESSION_METRICS = ["MAE", "RMSE", "R2"]
CLASSIFICATION_METRICS = ["Precision", "Recall", "F1", "Accuracy"]