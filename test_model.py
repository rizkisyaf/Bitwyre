#!/usr/bin/env python3

import torch
import numpy as np
import json
import os

def test_model(model_path, input_size=70):
    print(f'Testing model: {model_path}')
    
    # Check if metrics file exists
    metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print('\nModel Metrics:')
        print(f'  Accuracy: {metrics["accuracy"]:.4f}')
        print(f'  Precision: {metrics["precision"]:.4f}')
        print(f'  Recall: {metrics["recall"]:.4f}')
        print(f'  F1 Score: {metrics["f1_score"]:.4f}')
        print(f'  ROC AUC: {metrics["roc_auc"]:.4f}')
        print(f'  Confusion Matrix:')
        print(f'    TN: {metrics["confusion_matrix"][0][0]}, FP: {metrics["confusion_matrix"][0][1]}')
        print(f'    FN: {metrics["confusion_matrix"][1][0]}, TP: {metrics["confusion_matrix"][1][1]}')
    
    try:
        # Load the model
        model = torch.jit.load(model_path)
        model.eval()
        print('Model loaded successfully')
        
        # Create random input
        inputs = []
        for i in range(20):  # Test with 20 different inputs
            # Create input with some variation
            input_tensor = torch.randn(1, input_size)
            inputs.append(input_tensor)
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for input_tensor in inputs:
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
        
        # Print predictions
        print('\nModel Predictions:')
        for i, pred in enumerate(predictions):
            signal = 'BUY' if pred > 0.55 else 'SELL' if pred < 0.45 else 'NEUTRAL'
            confidence = abs(pred - 0.5) * 200  # Convert to percentage
            print(f'Input {i+1}: {pred:.4f} - {signal} [Confidence: {confidence:.2f}%]')
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        print(f'\nPrediction Statistics:')
        print(f'Mean: {mean_pred:.4f}')
        print(f'Std Dev: {std_pred:.4f}')
        print(f'Min: {min_pred:.4f}')
        print(f'Max: {max_pred:.4f}')
        
        # Count signals
        buy_count = sum(1 for p in predictions if p > 0.55)
        sell_count = sum(1 for p in predictions if p < 0.45)
        neutral_count = sum(1 for p in predictions if 0.45 <= p <= 0.55)
        
        print(f'\nSignal Distribution:')
        print(f'BUY: {buy_count} ({buy_count/len(predictions)*100:.1f}%)')
        print(f'SELL: {sell_count} ({sell_count/len(predictions)*100:.1f}%)')
        print(f'NEUTRAL: {neutral_count} ({neutral_count/len(predictions)*100:.1f}%)')
        
        # Check if predictions are meaningful
        if std_pred < 0.05:
            print('\n⚠️ WARNING: Model predictions have low variance!')
        else:
            print('\n✅ Model predictions have good variance')
        
        if abs(mean_pred - 0.5) > 0.1:
            print(f'⚠️ WARNING: Model predictions are biased towards {"BUY" if mean_pred > 0.5 else "SELL"}!')
        else:
            print('✅ Model predictions are balanced around 0.5')
        
        return True
    
    except Exception as e:
        print(f'Error testing model: {e}')
        return False

if __name__ == "__main__":
    import sys
    
    model_path = 'model_trained.pt'
    input_size = 70
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        input_size = int(sys.argv[2])
    
    test_model(model_path, input_size) 