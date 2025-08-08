#!/usr/bin/env python3
"""
Main entry point for the Predictive Maintenance ML Pipeline
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import PredictiveMaintenancePipeline
from src.predict import PredictiveMaintenancePredictor, create_sample_file
from src.config import *


def run_pipeline(tasks=None, dataset_id=None, failure_threshold=None):
    """
    Run the predictive maintenance pipeline
    
    Args:
        tasks (list): List of tasks to run ('regression', 'classification')
        dataset_id (str): Dataset ID to use
        failure_threshold (int): Failure threshold for classification
    """
    
    # Set defaults if not provided
    if tasks is None:
        tasks = ['regression', 'classification']
    if dataset_id is None:
        dataset_id = DATASET_ID
    if failure_threshold is None:
        failure_threshold = FAILURE_THRESHOLD
    
    print("="*60)
    print("PREDICTIVE MAINTENANCE ML PIPELINE")
    print("="*60)
    print(f"Dataset: {dataset_id}")
    print(f"Failure Threshold: {failure_threshold}")
    print(f"Tasks: {tasks}")
    print("="*60)
    
    # Initialize and run pipeline
    pipeline = PredictiveMaintenancePipeline(
        dataset_id=dataset_id,
        failure_threshold=failure_threshold
    )
    
    pipeline.run_pipeline(tasks=tasks)
    
    return pipeline


def run_prediction(input_file=None, dataset_id=None, failure_threshold=None, output_file=None):
    """
    Run prediction using trained models
    
    Args:
        input_file (str): Path to input file (optional, uses sample if None)
        dataset_id (str): Dataset ID to use
        failure_threshold (int): Failure threshold for classification
        output_file (str): Path to save results (optional)
    """
    
    # Set defaults
    if dataset_id is None:
        dataset_id = DATASET_ID
    if failure_threshold is None:
        failure_threshold = FAILURE_THRESHOLD
    
    print("="*60)
    print("PREDICTIVE MAINTENANCE PREDICTION")
    print("="*60)
    print(f"Dataset: {dataset_id}")
    print(f"Failure Threshold: {failure_threshold}")
    if input_file:
        print(f"Input File: {input_file}")
    else:
        print("Input: Sample data")
    print("="*60)
    
    # Initialize predictor
    predictor = PredictiveMaintenancePredictor(
        dataset_id=dataset_id,
        failure_threshold=failure_threshold
    )
    
    # Make predictions
    if input_file:
        if not os.path.exists(input_file):
            print(f"ERROR: File not found: {input_file}")
            return None
        
        predictions = predictor.predict_from_file(input_file)
        
        # Display summary
        print(f"\nPREDICTION SUMMARY:")
        print(f"Total samples: {len(predictions)}")
        
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
    else:
        # Create and use sample file
        sample_file = create_sample_file("temp_sample.txt")
        predictions = predictor.predict_from_file(sample_file)
        
        print("\nSAMPLE PREDICTION RESULTS:")
        print("="*40)
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
        
        # Clean up temp file
        if os.path.exists(sample_file):
            os.remove(sample_file)
    
    # Save results if requested
    if output_file and predictions:
        import json
        from datetime import datetime
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_id': dataset_id,
            'failure_threshold': failure_threshold,
            'predictions': predictions
        }
        
        if input_file:
            results['input_file'] = input_file
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print("="*60)
    return predictions


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance ML Pipeline and Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training Pipeline:
  python main.py                                    # Run full pipeline
  python main.py --mode train --tasks regression   # Run only regression
  python main.py --mode train --dataset FD002      # Custom dataset
  
  # Prediction:
  python main.py --mode predict --input data.txt   # Predict from file
  python main.py --mode predict --input data.txt --output results.json  # Save results
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['train', 'predict'],
        default='train',
        help='Mode to run: train (default) or predict'
    )
    
    # Training arguments
    parser.add_argument(
        '--tasks', 
        nargs='+', 
        choices=['regression', 'classification'],
        default=['regression', 'classification'],
        help='Tasks to run for training (default: both regression and classification)'
    )
    
    # Prediction arguments
    parser.add_argument(
        '--input',
        type=str,
        help='Input file for prediction (CSV or text file)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save prediction results (JSON format)'
    )
    
    # Common arguments
    parser.add_argument(
        '--dataset', 
        type=str, 
        default=DATASET_ID,
        help=f'Dataset ID to use (default: {DATASET_ID})'
    )
    
    parser.add_argument(
        '--threshold', 
        type=int, 
        default=FAILURE_THRESHOLD,
        help=f'Failure threshold for classification (default: {FAILURE_THRESHOLD})'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            # Run training pipeline
            pipeline = run_pipeline(
                tasks=args.tasks,
                dataset_id=args.dataset,
                failure_threshold=args.threshold
            )
            
            print("\n" + "="*60)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Models saved to: {MODELS_PATH}")
            print(f"Results saved to: {REPORTS_PATH}")
            print("="*60)
            
        elif args.mode == 'predict':
            # Run prediction
            predictions = run_prediction(
                input_file=args.input,
                dataset_id=args.dataset,
                failure_threshold=args.threshold,
                output_file=args.output
            )
            
            print("\n" + "="*60)
            print("PREDICTION COMPLETED SUCCESSFULLY!")
            print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Operation failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
