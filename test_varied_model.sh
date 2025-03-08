#!/bin/bash

# Script to test the varied model

echo "Starting varied model test..."

# Create a simple Python script to test the model
cat > test_model_prediction.py << 'EOF'
#!/usr/bin/env python3

import torch
import numpy as np

def test_model(model_path, input_size=25):
    """Test the model with random input"""
    print(f"Testing model: {model_path}")
    
    try:
        # Load the model
        model = torch.jit.load(model_path)
        model.eval()
        print("Model loaded successfully")
        
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
        print("\nModel Predictions:")
        for i, pred in enumerate(predictions):
            signal = "BUY" if pred > 0.55 else "SELL" if pred < 0.45 else "NEUTRAL"
            confidence = abs(pred - 0.5) * 200  # Convert to percentage
            print(f"Input {i+1}: {pred:.4f} - {signal} [Confidence: {confidence:.2f}%]")
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        print(f"\nPrediction Statistics:")
        print(f"Mean: {mean_pred:.4f}")
        print(f"Std Dev: {std_pred:.4f}")
        print(f"Min: {min_pred:.4f}")
        print(f"Max: {max_pred:.4f}")
        
        # Count signals
        buy_count = sum(1 for p in predictions if p > 0.55)
        sell_count = sum(1 for p in predictions if p < 0.45)
        neutral_count = sum(1 for p in predictions if 0.45 <= p <= 0.55)
        
        print(f"\nSignal Distribution:")
        print(f"BUY: {buy_count} ({buy_count/len(predictions)*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/len(predictions)*100:.1f}%)")
        print(f"NEUTRAL: {neutral_count} ({neutral_count/len(predictions)*100:.1f}%)")
        
        # Check if predictions are meaningful
        if std_pred < 0.05:
            print("\n⚠️ WARNING: Model predictions have low variance!")
        else:
            print("\n✅ Model predictions have good variance")
        
        if abs(mean_pred - 0.5) < 0.05:
            print("✅ Model predictions are balanced around 0.5")
        else:
            print(f"⚠️ WARNING: Model predictions are biased towards {'BUY' if mean_pred > 0.5 else 'SELL'}!")
        
        return True
    
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_model_prediction.py <model_path> [input_size]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_size = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    
    test_model(model_path, input_size)
EOF

# Make the script executable
chmod +x test_model_prediction.py

# Run the test
python3 test_model_prediction.py model_varied.pt 25

# Clean up
rm test_model_prediction.py

echo "Varied model test completed!" 