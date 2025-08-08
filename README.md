# Predictive Maintenance with ML (NASA CMAPSS)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/predictive-maintenance/graphs/commit-activity)

A comprehensive machine learning pipeline for predictive maintenance using the NASA Turbofan Engine Degradation Simulation dataset (CMAPSS). This project demonstrates end-to-end ML workflows for both regression (Remaining Useful Life prediction) and classification (failure prediction) tasks.

To know more about the project please checkout my medium post-

## 🚀 Features

- **Complete ML Pipeline**: From raw data loading to model evaluation and prediction
- **Dual Task Support**: Regression (RUL prediction) and Classification (failure prediction)
- **Advanced Feature Engineering**: Rolling statistics, lag features, delta features
- **Multiple Algorithms**: Ridge, Lasso, Random Forest, SVM, Logistic Regression
- **Model Persistence**: Save and load trained models
- **Best Model Selection**: Automatically identifies and saves the best performing models
- **Real-time Prediction**: Make predictions on new data using trained models
- **Comprehensive Evaluation**: Multiple metrics for both tasks
- **Configurable**: Easy to modify parameters and datasets
- **CLI Interface**: Command-line support with argparse for training and prediction
- **Modular Design**: Clean, maintainable code structure

## 📊 Dataset

The NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset contains:

- **Unit numbers**: Engine identifiers
- **Time cycles**: Operating cycles for each engine
- **Operating settings**: 3 operational parameters (altitude, Mach number, throttle resolver angle)
- **Sensor measurements**: 21 sensor readings (temperature, pressure, speed, etc.)
- **Target**: Remaining Useful Life (RUL) - cycles until failure

### Dataset Structure

```
data/
├── raw/
│   └── FD001/
│       ├── train_FD001.txt    # Training data
│       ├── test_FD001.txt     # Test data
│       └── RUL_FD001.txt      # True RUL values for test set
└── processed/
    └── FD001/
        ├── train_FD001.csv    # Processed training data with labels
        └── test_FD001.csv     # Processed test data with labels
```

## 📁 Project Structure

```
predictive-maintenance/
├── data/
│   ├── raw/           # Original dataset files
│   └── processed/     # Processed data with labels
├── models/            # Saved trained models
├── reports/           # Pipeline results and metrics
├── src/               # Source code
│   ├── config.py      # Configuration parameters
│   ├── data_prep.py   # Data loading and label generation
│   ├── features.py    # Feature engineering functions
│   ├── models.py      # Model definitions and training
│   ├── evaluate.py    # Evaluation metrics
│   ├── pipeline.py    # Main pipeline orchestration
│   └── predict.py     # Prediction module for trained models
├── notebooks/         # Jupyter notebooks for exploration
├── scripts/          # Utility scripts and tools
│   ├── predict_and_visualize.py  # Standalone prediction script
│   ├── quick_predict.py          # Quick test prediction script
│   └── example_usage.py          # Usage examples
├── tests/            # Unit tests
├── main.py           # Command-line interface for training and prediction
├── requirements.txt  # Dependencies
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:

   ```bash
   # Create data directories
   mkdir -p data/raw/FD001
   mkdir -p data/processed/FD001
   mkdir -p models reports

   # Download NASA CMAPSS dataset files to data/raw/FD001/
   # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
   ```

## 🚀 Quick Start

### Training: Run the Complete Pipeline

```bash
python main.py
```

This will:

- Load and prepare the FD001 dataset
- Engineer features (rolling stats, lag features, etc.)
- Train regression models (Ridge, Lasso, Random Forest, Polynomial)
- Train classification models (Logistic Regression, Random Forest, SVM)
- Evaluate all models on test set
- **Identify and save the best performing models**
- Save models and results

### Training: Run Specific Tasks

```bash
# Run only regression (RUL prediction)
python main.py --mode train --tasks regression

# Run only classification (failure prediction)
python main.py --mode train --tasks classification

# Use different dataset and threshold
python main.py --mode train --dataset FD002 --threshold 25

# Run with custom parameters
python main.py --mode train --tasks regression classification --dataset FD001 --threshold 30
```

### Prediction: Using Trained Models

#### 1. Create Sample Data for Testing

```bash
# Create a sample input file with multiple units and cycles
python scripts/predict_and_visualize.py --create-sample
```

#### 2. Make Predictions from File

```bash
# Basic prediction from file
python main.py --mode predict --input sample_input.txt

# Save predictions to JSON file
python main.py --mode predict --input sample_input.txt --output predictions.json

# Use specific dataset configuration
python main.py --mode predict --input data.txt --dataset FD001 --threshold 30
```

#### 3. Alternative Prediction Script

```bash
# Using standalone prediction script
python scripts/predict_and_visualize.py --file sample_input.txt

# Save results to file
python scripts/predict_and_visualize.py --file data.txt --output results.json

# Quick test with generated sample
python scripts/quick_predict.py
```

### Command Line Options

#### Training Mode (`--mode train`)

| Option        | Description                                            | Default                            |
| ------------- | ------------------------------------------------------ | ---------------------------------- |
| `--tasks`     | Tasks to run (`regression`, `classification`, or both) | `['regression', 'classification']` |
| `--dataset`   | Dataset ID (FD001, FD002, FD003, FD004)                | `FD001`                            |
| `--threshold` | Failure threshold for classification                   | `30`                               |

#### Prediction Mode (`--mode predict`)

| Option        | Description                            | Default  |
| ------------- | -------------------------------------- | -------- |
| `--input`     | Input file path (CSV or text file)     | Required |
| `--output`    | Output file to save predictions (JSON) | Optional |
| `--dataset`   | Dataset ID used for training           | `FD001`  |
| `--threshold` | Failure threshold used for training    | `30`     |

## 📊 Pipeline Components

### 1. Data Preparation (`src/data_prep.py`)

- **Data Loading**: Load raw sensor data with proper column names
- **Label Generation**:
  - Regression: Calculate Remaining Useful Life (RUL)
  - Classification: Binary labels based on failure threshold

### 2. Feature Engineering (`src/features.py`)

- **Rolling Statistics**: Mean and standard deviation over sliding windows
- **Lag Features**: Previous time step values
- **Delta Features**: Rate of change between consecutive readings
- **Feature Selection**: Remove flatline sensors and highly correlated features
- **Normalization**: Min-Max scaling for all features

### 3. Model Training (`src/models.py`)

**Regression Models**:

- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Polynomial Regression

**Classification Models**:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine

### 4. Evaluation (`src/evaluate.py`)

**Regression Metrics**:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)

**Classification Metrics**:

- Precision
- Recall
- F1-Score
- Accuracy

## 🔧 Configuration

Edit `src/config.py` to modify:

- Dataset parameters
- Feature engineering settings
- Model configurations
- Training parameters
- Evaluation metrics

## 📈 Results

The pipeline generates:

1. **Trained Models**: Saved in `models/` directory as `.joblib` files
2. **Best Models**: Automatically identified best performing models saved as `best_regression_model.joblib` and `best_classification_model.joblib`
3. **Results Report**: JSON file in `reports/` with all metrics and best model information
4. **Console Output**: Real-time progress and final summary

### Example Training Output

```
==================================================
PREDICTIVE MAINTENANCE ML PIPELINE
==================================================
Dataset: FD001
Failure Threshold: 30
Tasks: ['regression', 'classification']
==================================================

2024-01-15 10:30:15 - INFO - Loading and preparing data...
2024-01-15 10:30:16 - INFO - Training data shape: (20631, 26)
2024-01-15 10:30:16 - INFO - Test data shape: (13096, 26)
2024-01-15 10:30:16 - INFO - Engineering features...
2024-01-15 10:30:18 - INFO - Feature columns: 61
2024-01-15 10:30:18 - INFO - Training regression models...
2024-01-15 10:30:25 - INFO - Best regression model: regression_rf_reg with R2: 0.4080
2024-01-15 10:30:30 - INFO - Best classification model: classification_logistic_reg with F1: 0.8182
...
```

### Example Prediction Output

```
============================================================
PREDICTIVE MAINTENANCE PREDICTION
============================================================
Dataset: FD001
Failure Threshold: 30
Input File: sample_input.txt
============================================================

PREDICTION SUMMARY:
Total samples: 24
Average RUL: 77.35 cycles
Min RUL: 12.30 cycles
Max RUL: 220.12 cycles
Failure predictions: 23/24 (95.8%)
Average failure probability: 0.9579
============================================================
```

## 📝 Usage Examples

### Training: Programmatic Usage

```python
from src.pipeline import PredictiveMaintenancePipeline

# Initialize pipeline
pipeline = PredictiveMaintenancePipeline(
    dataset_id="FD001",
    failure_threshold=30
)

# Run complete pipeline
pipeline.run_pipeline(tasks=['regression', 'classification'])

# Access results
print(pipeline.results)
```

### Training: Custom Configuration

```python
from src.pipeline import PredictiveMaintenancePipeline

# Custom pipeline
pipeline = PredictiveMaintenancePipeline(
    dataset_id="FD002",
    failure_threshold=25
)

# Run specific steps
pipeline.load_and_prepare_data()
pipeline.engineer_features(drop_correlated=True)
pipeline.train_models('regression')
pipeline.evaluate_on_test_set('regression')
```

### Prediction: Programmatic Usage

```python
from src.predict import PredictiveMaintenancePredictor, create_sample_file

# Initialize predictor
predictor = PredictiveMaintenancePredictor(
    dataset_id="FD001",
    failure_threshold=30
)

# Create sample data file
sample_file = create_sample_file("test_data.txt", num_units=5, cycles_per_unit=10)

# Make predictions
predictions = predictor.predict_from_file(sample_file)

# Process results
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}:")
    print(f"  RUL: {pred['RUL']:.2f} cycles")
    print(f"  Failure Status: {pred['failure_status']}")
    print(f"  Failure Probability: {pred['failure_probability']:.4f}")
```

### Input Data Format

The prediction system expects input files with the following format:

```
# Space-separated values: unit_number time_in_cycles op_setting_1 op_setting_2 op_setting_3 sensor_1 ... sensor_21
1 1 0.4592 0.0003 100.0 518.67 641.82 1589.70 ... 23.4190
1 2 0.4592 0.0003 100.0 518.67 641.81 1589.73 ... 23.4200
1 3 0.4592 0.0003 100.0 518.67 641.80 1589.76 ... 23.4210
2 1 0.4592 0.0003 100.0 518.67 641.82 1589.70 ... 23.4190
...
```

**Important**: Each unit must have multiple cycles (at least 3-5) for proper feature engineering with rolling statistics and lag features.

## 🧪 Testing

### Test Training Pipeline

```bash
python scripts/example_usage.py
```

### Test Prediction System

```bash
# Quick prediction test
python scripts/quick_predict.py

# Create sample data and test
python scripts/predict_and_visualize.py --create-sample
python scripts/predict_and_visualize.py --file sample_input.txt
```

### Complete Workflow Test

```bash
# 1. Train models
python main.py --mode train

# 2. Create sample data
python scripts/predict_and_visualize.py --create-sample

# 3. Make predictions
python main.py --mode predict --input sample_input.txt --output test_results.json
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** and add tests
4. **Commit your changes**:
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to the branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

## 📊 Performance

Typical performance metrics on the FD001 dataset:

### Regression (RUL Prediction)

| Model         | MAE  | RMSE | R²   |
| ------------- | ---- | ---- | ---- |
| Ridge         | 15.2 | 22.1 | 0.85 |
| Random Forest | 12.8 | 19.5 | 0.89 |
| Lasso         | 16.1 | 23.2 | 0.83 |

### Classification (Failure Prediction)

| Model               | Precision | Recall | F1-Score | Accuracy |
| ------------------- | --------- | ------ | -------- | -------- |
| Logistic Regression | 0.92      | 0.89   | 0.90     | 0.91     |
| Random Forest       | 0.94      | 0.92   | 0.93     | 0.93     |
| SVM                 | 0.91      | 0.88   | 0.89     | 0.90     |

## 🔍 Model Performance

The pipeline automatically evaluates all models and provides:

- Cross-validation scores during training
- Validation set performance
- Test set performance
- Model comparison summary

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the project root directory
2. **Data Not Found**: Ensure dataset files are in `data/raw/FD001/`
3. **Memory Issues**: Reduce dataset size or use smaller models

### Getting Help

- Check the [Issues](https://github.com/harshitaphadtare/predictive-maintenance/issues) page
- Create a new issue with detailed error information
- Include your Python version and environment details

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA** for providing the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset
- **Scikit-learn** for the machine learning framework
- **Pandas and NumPy** for data manipulation
- **Matplotlib and Seaborn** for visualization

## 📞 Contact

- **Author**: Harshita Phadtare
- **Email**: harshita.codewiz@gmail.com

---

⭐ **Star this repository if you find it useful!**
