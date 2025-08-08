#!/usr/bin/env python3
"""
Example usage of the Predictive Maintenance ML Pipeline
"""

import sys
import os

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from pipeline import PredictiveMaintenancePipeline
from config import *


def example_1_basic_pipeline():
    """Example 1: Run the complete pipeline"""
    print("="*60)
    print("EXAMPLE 1: Basic Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PredictiveMaintenancePipeline()
    
    # Run complete pipeline
    pipeline.run_pipeline()
    
    print("Pipeline completed!")
    return pipeline


def example_2_custom_configuration():
    """Example 2: Custom configuration"""
    print("="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Custom pipeline with different parameters
    pipeline = PredictiveMaintenancePipeline(
        dataset_id="FD001",
        failure_threshold=25  # Different threshold
    )
    
    # Run only regression
    pipeline.run_pipeline(tasks=['regression'])
    
    print("Custom pipeline completed!")
    return pipeline


def example_3_step_by_step():
    """Example 3: Step-by-step pipeline execution"""
    print("="*60)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PredictiveMaintenancePipeline()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    pipeline.load_and_prepare_data()
    
    # Step 2: Engineer features
    print("Step 2: Engineering features...")
    pipeline.engineer_features(drop_correlated=True)
    
    # Step 3: Train regression models
    print("Step 3: Training regression models...")
    pipeline.train_models('regression')
    
    # Step 4: Evaluate on test set
    print("Step 4: Evaluating on test set...")
    test_results = pipeline.evaluate_on_test_set('regression')
    
    # Step 5: Save models and results
    print("Step 5: Saving models and results...")
    pipeline.save_models()
    pipeline.save_results()
    
    print("Step-by-step pipeline completed!")
    return pipeline


def example_4_model_comparison():
    """Example 4: Compare model performance"""
    print("="*60)
    print("EXAMPLE 4: Model Comparison")
    print("="*60)
    
    # Run pipeline
    pipeline = PredictiveMaintenancePipeline()
    pipeline.run_pipeline(tasks=['regression', 'classification'])
    
    # Compare regression models
    print("\nREGRESSION MODEL COMPARISON:")
    if 'regression_test_results' in pipeline.results:
        for model_name, metrics in pipeline.results['regression_test_results'].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Compare classification models
    print("\nCLASSIFICATION MODEL COMPARISON:")
    if 'classification_test_results' in pipeline.results:
        for model_name, metrics in pipeline.results['classification_test_results'].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    return pipeline


def example_5_feature_analysis():
    """Example 5: Analyze feature importance"""
    print("="*60)
    print("EXAMPLE 5: Feature Analysis")
    print("="*60)
    
    # Run pipeline
    pipeline = PredictiveMaintenancePipeline()
    pipeline.load_and_prepare_data()
    pipeline.engineer_features()
    
    # Get feature information
    print(f"Total features: {len(pipeline.feature_columns)}")
    print(f"Feature columns: {pipeline.feature_columns[:10]}...")  # Show first 10
    
    # Train a model and get feature importance
    X_train, X_val, y_train, y_val, X, y = pipeline.prepare_training_data('regression')
    
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = rf_model.feature_importances_
    feature_names = pipeline.feature_columns
    
    # Sort by importance
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(importance_pairs[:10]):
        print(f"{i+1:2d}. {feature}: {importance:.4f}")
    
    return pipeline


def main():
    """Run all examples"""
    print("PREDICTIVE MAINTENANCE PIPELINE EXAMPLES")
    print("="*60)
    
    try:
        # Run examples
        example_1_basic_pipeline()
        print("\n" + "="*60 + "\n")
        
        example_2_custom_configuration()
        print("\n" + "="*60 + "\n")
        
        example_3_step_by_step()
        print("\n" + "="*60 + "\n")
        
        example_4_model_comparison()
        print("\n" + "="*60 + "\n")
        
        example_5_feature_analysis()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Example failed - {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 