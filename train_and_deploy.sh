#!/bin/bash

# Script to train and deploy the trading model

# Set paths
DATA_DIR="data"
PYTHON_DIR="python"
MODEL_DIR="models"
DATA_FILE="${DATA_DIR}/futures_orderbook_data.csv"
MODEL_FILE="model_trained.pt"
MEAN_FILE="mean.npy"
STD_FILE="std.npy"

# Create directories if they don't exist
mkdir -p ${DATA_DIR}
mkdir -p ${MODEL_DIR}

# Check if data file exists
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ Data file ${DATA_FILE} not found. Please run data collection first."
    echo "You can collect data using:"
    echo "./build/src/collect_orderbook_data btcusdt 1440 ${DATA_FILE}"
    exit 1
fi

echo "🔍 Starting model training and deployment pipeline..."

# Train the model on real data
echo "🧠 Training model on real market data..."
python3 ${PYTHON_DIR}/training/train_model.py --data ${DATA_FILE} \
  --output ${MODEL_FILE} \
  --mean ${MEAN_FILE} \
  --std ${STD_FILE} \
  --epochs 100 \
  --future-window 20 \
  --price-change-threshold 0.0008 \
  --neutral-handling distribute \
  --use-class-weights \
  --batch-size 64 \
  --learning-rate 0.0005

# Check if training was successful
if [ ! -f "${MODEL_FILE}" ]; then
    echo "❌ Model training failed. Please check the logs for errors."
    exit 1
fi

# Create a test script to verify the model
echo "🧪 Creating test script for the trained model..."
cat > test_trained_model.py << 'EOF'
#!/usr/bin/env python3

import torch
import numpy as np
import json
import os
import sys

def test_model(model_path, input_size=61):
    """Test the trained model with random input"""
    print(f"Testing model: {model_path}")
    
    # Check if metrics file exists
    metrics_path = os.path.splitext(model_path)[0] + "_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print("\nModel Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
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
        
        if abs(mean_pred - 0.5) > 0.1:
            print(f"⚠️ WARNING: Model predictions are biased towards {'BUY' if mean_pred > 0.5 else 'SELL'}!")
        else:
            print("✅ Model predictions are balanced around 0.5")
        
        return True
    
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_trained_model.py <model_path> [input_size]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_size = int(sys.argv[2]) if len(sys.argv) > 2 else 61
    
    test_model(model_path, input_size)
EOF

# Make the script executable
chmod +x test_trained_model.py

# Test the trained model
echo "🧪 Testing the trained model..."
python3 test_trained_model.py ${MODEL_FILE}

# Clean up
rm test_trained_model.py

# Update the model path in main.cpp
echo "📝 Updating model path in main.cpp..."
MODEL_PATH_ESCAPED=$(echo ${MODEL_FILE} | sed 's/\//\\\//g')
sed -i.bak "s/model_varied.pt/${MODEL_PATH_ESCAPED}/g" src/main.cpp
rm src/main.cpp.bak

echo "✅ Training and deployment complete!"
echo "The new model is ready to use with the trading bot."
echo "The model path in main.cpp has been updated to use '${MODEL_FILE}'." 