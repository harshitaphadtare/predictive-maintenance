#!/usr/bin/env python3
"""
Quick prediction script for testing the predictive maintenance system
"""

import sys
import os

# Add src directory to path (go up one level since we're in scripts/)
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from predict import PredictiveMaintenancePredictor, create_sample_file


def main():
    """Quick test of the prediction system"""
    
    print("="*60)
    print("QUICK PREDICTION TEST")
    print("="*60)
    
    try:
        # Initialize predictor
        print("Loading models and configuration...")
        predictor = PredictiveMaintenancePredictor()
        
        # Create sample file
        print("Creating sample data file...")
        sample_file = create_sample_file("test_sample.txt", num_units=2, cycles_per_unit=5)
        
        # Make prediction
        print("Making predictions...")
        predictions = predictor.predict_from_file(sample_file)
        
        # Display results
        print("\n" + "="*40)
        print("PREDICTION RESULTS")
        print("="*40)
        print(f"Total predictions: {len(predictions)}")
        
        if predictions:
            rul_values = [p.get('RUL', 0) for p in predictions if 'RUL' in p]
            failure_predictions = [p.get('failure_prediction', 0) for p in predictions if 'failure_prediction' in p]
            failure_probs = [p.get('failure_probability', 0) for p in predictions if 'failure_probability' in p]
            
            print("\nPrediction Summary:")
            if rul_values:
                print(f"  Average RUL: {sum(rul_values)/len(rul_values):.2f} cycles")
                print(f"  Min RUL: {min(rul_values):.2f} cycles")
                print(f"  Max RUL: {max(rul_values):.2f} cycles")
            
            if failure_predictions:
                failure_count = sum(failure_predictions)
                print(f"  Failure predictions: {failure_count}/{len(failure_predictions)} ({failure_count/len(failure_predictions)*100:.1f}%)")
            
            if failure_probs:
                avg_failure_prob = sum(failure_probs)/len(failure_probs)
                print(f"  Average failure probability: {avg_failure_prob:.4f}")
        
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)
        
        print("="*60)
        print("✅ Prediction test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        print("="*60)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 