#!/bin/bash

# Script to collect futures orderbook data

# Set paths
DATA_DIR="data"
DATA_FILE="${DATA_DIR}/futures_orderbook_data.csv"

# Create data directory if it doesn't exist
mkdir -p ${DATA_DIR}

# Check if the data collection executable exists
if [ ! -f "build/src/collect_orderbook_data" ]; then
    echo "❌ Data collection executable not found. Please build the project first."
    echo "You can build the project using:"
    echo "mkdir -p build && cd build && cmake .. && make -j4"
    exit 1
fi

# Collect data
echo "📊 Starting futures orderbook data collection..."
echo "This will collect data for 24 hours (1440 minutes)."
echo "You can stop the collection at any time by pressing Ctrl+C."
echo "The data will be saved to ${DATA_FILE}"
echo ""

# Run the data collection
./build/src/collect_orderbook_data btcusdt 1440 ${DATA_FILE}

echo "✅ Data collection complete!"
echo "You can now train the model using:"
echo "./train_and_deploy.sh" 