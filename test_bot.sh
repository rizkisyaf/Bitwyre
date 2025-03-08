#!/bin/bash

# Script to test the trading bot with placeholder API keys

echo "Starting trading bot test (will run for 10 seconds)..."

# Run the trading bot with test connection parameters
./build/src/trading_bot btcusdt 10 placeholder_api_key placeholder_secret_key --test_connection

echo "Test completed!" 