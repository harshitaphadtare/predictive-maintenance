#!/usr/bin/env python3
"""
Prediction script for Predictive Maintenance system
"""

import argparse
import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add src directory to path (go up one level since we're in scripts/)
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from predict import PredictiveMaintenancePredictor, create_sample_file
from config import *


def predict_from_file(file_path, dataset_id=None, failure_threshold=None, output_file=None):
    """
    Make predictions from a file
    
    Args:
        file_path: Path to input file
        dataset_id: Dataset ID to use
        failure_threshold: Failure threshold for classification
        output_file: Path to save results (optional)
    """
    
    # Set defaults
    if dataset_id is None:
        dataset_id = DATASET_ID
    if failure_threshold is None:
        failure_threshold = FAILURE_THRESHOLD
    
    print("="*60)
    print("FILE-BASED PREDICTION")
    print("="*60)
    print(f"Input File: {file_path}")
    print(f"Dataset: {dataset_id}")
    print(f"Failure Threshold: {failure_threshold}")
    print("="*60)
    
    # Initialize predictor
    predictor = PredictiveMaintenancePredictor(
        dataset_id=dataset_id,
        failure_threshold=failure_threshold
    )
    
    # Make predictions
    predictions = predictor.predict_from_file(file_path)
    
    # Display summary
    print(f"\nPREDICTION SUMMARY:")
    print(f"Total samples: {len(predictions)}")
    
    if predictions:
        rul_values = [p.get('RUL', 0) for p in predictions if 'RUL' in p]
        failure_predictions = [p.get('failure_prediction', 0) for p in predictions if 'failure_prediction' in p]
        failure_probs = [p.get('failure_probability', 0) for p in predictions if 'failure_probability' in p]
        
        if rul_values:
            print(f"Average RUL: {sum(rul_values)/len(rul_values):.2f} cycles")
            print(f"Min RUL: {min(rul_values):.2f} cycles")
            print(f"Max RUL: {max(rul_values):.2f} cycles")
        
        if failure_predictions:
            failure_count = sum(failure_predictions)
            print(f"Failure predictions: {failure_count}/{len(failure_predictions)} ({failure_count/len(failure_predictions)*100:.1f}%)")
        
        if failure_probs:
            avg_failure_prob = sum(failure_probs)/len(failure_probs)
            print(f"Average failure probability: {avg_failure_prob:.4f}")
    
    # Save results if requested
    if output_file:
        results = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'dataset_id': dataset_id,
            'failure_threshold': failure_threshold,
            'predictions': predictions
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print("="*60)
    return predictions





def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_and_visualize.py --file input.txt            # Predict from file
  python predict_and_visualize.py --file input.txt --output results.json  # Save results
  python predict_and_visualize.py --create-sample             # Create sample input file
  python predict_and_visualize.py --dataset FD002 --threshold 25  # Custom parameters
        """
    )
    
    # Prediction modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--file', 
        type=str,
        help='Path to input file (CSV or text file)'
    )
    group.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample input file for testing'
    )
    
    # Optional arguments
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
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save results (JSON format)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.create_sample:
            # Create sample file
            sample_file = create_sample_file()
            print(f"\nSample file created: {sample_file}")
            print("You can use this file with: python predict_and_visualize.py --file sample_input.txt")
            
        elif args.file:
            # File-based prediction
            if not os.path.exists(args.file):
                print(f"ERROR: File not found: {args.file}")
                sys.exit(1)
            
            predict_from_file(
                file_path=args.file,
                dataset_id=args.dataset,
                failure_threshold=args.threshold,
                output_file=args.output
            )
        
    except Exception as e:
        print(f"\nERROR: Prediction failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 