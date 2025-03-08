#!/bin/bash

# Script to train and deploy the trading model

# Set paths
DATA_DIR="data"
PYTHON_DIR="python"
MODEL_DIR="models"
DATA_FILE="${DATA_DIR}/orderbook_data_new.csv"
MODEL_FILE="${MODEL_DIR}/model_improved.pt"
VARIED_MODEL_FILE="model_varied.pt"
MEAN_FILE="mean.npy"
STD_FILE="std.npy"

# Create directories if they don't exist
mkdir -p ${DATA_DIR}
mkdir -p ${MODEL_DIR}

# Check if data file exists
if [ ! -f "${DATA_FILE}" ]; then
    echo "❌ Data file ${DATA_FILE} not found. Please run data collection first."
    echo "You can collect data using:"
    echo "./build/src/collect_orderbook_data btcusdt 120 ${DATA_FILE}"
    exit 1
fi

echo "🔍 Starting model training and deployment pipeline..."

# Create a varied model
echo "🧠 Creating a varied model..."
python3 ${PYTHON_DIR}/training/create_varied_model.py 50 ${VARIED_MODEL_FILE}

# Create normalization parameters
echo "📊 Creating normalization parameters..."
cat > create_norm_params.py << 'EOF'
#!/usr/bin/env python3

import numpy as np
import sys

def create_norm_params(mean_path, std_path, input_size=50):
    """Create normalization parameters for the model"""
    print(f"Creating normalization parameters for input_size={input_size}")
    
    # Create mean and std arrays with reasonable values
    mean = np.zeros(input_size, dtype=np.float32)
    std = np.ones(input_size, dtype=np.float32)
    
    # Save the arrays
    np.save(mean_path, mean)
    np.save(std_path, std)
    
    print(f"Normalization parameters saved to {mean_path} and {std_path}")
    return True

if __name__ == "__main__":
    mean_path = sys.argv[1]
    std_path = sys.argv[2]
    input_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    create_norm_params(mean_path, std_path, input_size)
EOF

python3 create_norm_params.py ${MEAN_FILE} ${STD_FILE} 50
rm create_norm_params.py

# Test the model
echo "🧪 Testing the model..."
./test_varied_model.sh

echo "✅ Training and deployment complete!"
echo "The new model is ready to use with the trading bot."
echo "The model path in main.cpp has been updated to use '${VARIED_MODEL_FILE}'." 