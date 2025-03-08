#!/bin/bash

# Check if API key and secret key are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <api_key> <secret_key> [--spot]"
    echo "Options:"
    echo "  --spot    Use spot trading instead of USDM futures"
    exit 1
fi

API_KEY=$1
SECRET_KEY=$2
shift 2

USE_SPOT=""

# Parse additional options
while [ $# -gt 0 ]; do
    case "$1" in
        --spot)
            USE_SPOT="--spot"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "TESTING BINANCE API CONNECTION"
echo "========================================="
echo "API Key: ${API_KEY:0:4}...${API_KEY: -4}"
echo "Mode: $([ -n "$USE_SPOT" ] && echo "Spot" || echo "USDM Futures")"
echo "-----------------------------------------"

# Run only the test connection command
./build/src/trading_bot \
    --api-key "$API_KEY" \
    --secret-key "$SECRET_KEY" \
    $USE_SPOT \
    --test-connection

TEST_RESULT=$?
echo "-----------------------------------------"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Connection test PASSED"
else
    echo "❌ Connection test FAILED"
fi
echo "========================================="
exit $TEST_RESULT 