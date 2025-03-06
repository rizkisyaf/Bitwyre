#!/bin/bash

# Default parameters
SYMBOL="btcusdt"
DURATION=60
INITIAL_BALANCE=100.0
STOP_LOSS=0.5
MAX_DRAWDOWN=5.0
EXCHANGE_URL="wss://stream.binance.com:9443/ws"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --balance)
      INITIAL_BALANCE="$2"
      shift 2
      ;;
    --stop-loss)
      STOP_LOSS="$2"
      shift 2
      ;;
    --max-drawdown)
      MAX_DRAWDOWN="$2"
      shift 2
      ;;
    --exchange-url)
      EXCHANGE_URL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --symbol SYMBOL       Trading symbol (default: btcusdt)"
      echo "  --duration SECONDS    Test duration in seconds (default: 60)"
      echo "  --balance AMOUNT      Initial balance (default: 100.0)"
      echo "  --stop-loss PCT       Stop loss percentage (default: 0.5%)"
      echo "  --max-drawdown PCT    Max drawdown percentage (default: 5.0%)"
      echo "  --exchange-url URL    Exchange WebSocket URL (default: wss://stream.binance.com:9443/ws)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running performance test with:"
echo "  Symbol: $SYMBOL"
echo "  Duration: $DURATION seconds"
echo "  Initial balance: $INITIAL_BALANCE"
echo "  Stop loss: $STOP_LOSS%"
echo "  Max drawdown: $MAX_DRAWDOWN%"
echo "  Exchange URL: $EXCHANGE_URL"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Create the PyTorch model
echo "Creating PyTorch model..."
python3 create_model.py

# Set the DYLD_LIBRARY_PATH for macOS
export DYLD_LIBRARY_PATH="$PWD/venv/lib/python3.13/site-packages/torch/lib"

# Run the trading bot and capture output
echo "Running trading bot..."
./build/src/trading_bot --symbol $SYMBOL --exchange-url $EXCHANGE_URL --duration $DURATION --balance $INITIAL_BALANCE --stop-loss $STOP_LOSS --max-drawdown $MAX_DRAWDOWN > performance_results.txt 2>&1

# Display the last performance metrics
echo "Final performance metrics:"
grep -A 7 "Performance metrics:" performance_results.txt | tail -7

# Display the last USD volume metrics
echo "Final USD volume metrics:"
if grep -q "USD Volume metrics:" performance_results.txt; then
  grep -A 4 "USD Volume metrics:" performance_results.txt | tail -4
fi

# Display the last P&L metrics
echo "Final P&L metrics:"
if grep -q "P&L metrics:" performance_results.txt; then
  grep -A 3 "P&L metrics:" performance_results.txt | tail -3
fi

# Display uncapped P&L metrics for analysis
echo "Final Uncapped P&L metrics (for analysis):"
if grep -q "Uncapped P&L metrics" performance_results.txt; then
  grep -A 2 "Uncapped P&L metrics" performance_results.txt | tail -2
fi

# Display detailed P&L metrics if available
if grep -q "Detailed P&L metrics:" performance_results.txt; then
  echo "Detailed P&L metrics:"
  grep -A 9 "Detailed P&L metrics:" performance_results.txt | tail -9
fi

echo "Performance test completed. Full results saved to performance_results.txt" 