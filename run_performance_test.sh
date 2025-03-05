#!/bin/bash

# Activate the virtual environment
source venv/bin/activate 2>/dev/null || echo "No virtual environment found, continuing without it"

# Generate the PyTorch model
echo "Generating PyTorch model..."
python create_model.py

# Set the DYLD_LIBRARY_PATH for macOS
export DYLD_LIBRARY_PATH="$PWD/venv/lib/python3.13/site-packages/torch/lib"

# Run the trading bot with Binance connection
echo "Running trading bot with Binance connection..."
cd build

# Create an empty performance results file
touch ../performance_results.txt

# Run the trading bot in the background
./src/trading_bot "wss://stream.binance.com:9443/ws/btcusdt@depth" "btcusdt" "../model.pt" > ../performance_results.txt 2>&1 &
BOT_PID=$!

# Wait for 60 seconds
echo "Collecting performance metrics for 60 seconds..."
sleep 60

# Kill the trading bot process
echo "Stopping the trading bot..."
kill -SIGINT $BOT_PID 2>/dev/null
sleep 2
kill -9 $BOT_PID 2>/dev/null

# Extract and analyze performance metrics
echo "Performance test completed. Results saved to performance_results.txt"

# Display the last performance metrics
echo "Final performance metrics:"
grep -A 7 "Performance metrics:" ../performance_results.txt | tail -8 