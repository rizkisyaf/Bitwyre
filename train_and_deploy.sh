#!/bin/bash

# Script to train and deploy the trading model

# Set paths
DATA_DIR="data"
PYTHON_DIR="python"
MODEL_DIR="models"
DATA_FILE="${DATA_DIR}/orderbook_data_new.csv"
MODEL_FILE="${MODEL_DIR}/model_improved.pt"
MEAN_FILE="${MODEL_DIR}/mean_improved.npy"
STD_FILE="${MODEL_DIR}/std_improved.npy"

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

echo "🔍 Starting model training and testing pipeline..."

# Train the model
echo "🧠 Training the model..."
python3 ${PYTHON_DIR}/training/improved_training.py \
    --data ${DATA_FILE} \
    --model_output ${MODEL_FILE} \
    --mean_output ${MEAN_FILE} \
    --std_output ${STD_FILE} \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --prediction_horizon 5

# Check if training was successful
if [ ! -f "${MODEL_FILE}" ]; then
    echo "❌ Model training failed. Model file not created."
    exit 1
fi

# Test the model
echo "🧪 Testing the model..."
python3 ${PYTHON_DIR}/training/test_model.py \
    --model ${MODEL_FILE} \
    --data ${DATA_FILE} \
    --mean ${MEAN_FILE} \
    --std ${STD_FILE} \
    --test_size 100

# Convert the model for C++ usage
echo "🔄 Converting the model for C++ usage..."
python3 ${PYTHON_DIR}/training/convert_model.py convert \
    --input ${MODEL_FILE} \
    --output ${MODEL_DIR}/model_cpp.pt

# Copy files to the main directory for the trading bot
echo "📋 Copying files for the trading bot..."
cp ${MODEL_DIR}/model_cpp.pt model_new.pt
cp ${MEAN_FILE} mean.npy
cp ${STD_FILE} std.npy

echo "✅ Training and deployment complete!"
echo "The new model is ready to use with the trading bot."
echo "To use the new model, update the model path in main.cpp to 'model_new.pt'" 