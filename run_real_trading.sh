#!/bin/bash

# Check if API key and secret key are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <api_key> <secret_key> [options]"
    echo "Options:"
    echo "  --symbol SYMBOL       Trading symbol (default: btcusdt)"
    echo "  --duration SECONDS    Test duration in seconds (default: 3600)"
    echo "  --balance AMOUNT      Initial balance (default: 100.0)"
    echo "  --stop-loss PCT       Stop loss percentage (default: 0.5%)"
    echo "  --max-drawdown PCT    Max drawdown percentage (default: 5.0%)"
    echo "  --spot                Use spot trading instead of USDM futures"
    echo "  --leverage NUM        Set leverage for futures trading (1-125, default: 5)"
    echo "  --cross-margin        Use cross margin instead of isolated margin for futures"
    echo "  --test-connection     Only test API connection and exit"
    exit 1
fi

API_KEY=$1
SECRET_KEY=$2
shift 2

# Set default values
SYMBOL="btcusdt"
DURATION=3600  # 1 hour by default
BALANCE=100.0
STOP_LOSS=0.5
MAX_DRAWDOWN=5.0
USE_SPOT=""
LEVERAGE="5"
MARGIN_TYPE=""
TEST_CONNECTION=""

# Parse additional options
while [ $# -gt 0 ]; do
    case "$1" in
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --balance)
            BALANCE="$2"
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
        --spot)
            USE_SPOT="--spot"
            shift
            ;;
        --leverage)
            LEVERAGE="$2"
            shift 2
            ;;
        --cross-margin)
            MARGIN_TYPE="--cross-margin"
            shift
            ;;
        --test-connection)
            TEST_CONNECTION="--test-connection"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If test connection only, run in foreground
if [ -n "$TEST_CONNECTION" ]; then
    echo "========================================="
    echo "TESTING API CONNECTION ONLY"
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
fi

# Function to handle emergency stop
emergency_stop() {
    echo "Emergency stop requested. Canceling all orders and stopping the bot..."
    if [ -n "$BOT_PID" ]; then
        kill -SIGUSR1 $BOT_PID
    fi
    exit 1
}

# Set up trap for emergency stop
trap emergency_stop INT TERM

# Display warning for real trading
echo "========================================="
echo "⚠️  WARNING: STARTING REAL TRADING ⚠️"
echo "========================================="
echo "Symbol: $SYMBOL"
echo "Duration: $DURATION seconds"
echo "Stop Loss: $STOP_LOSS%"
echo "Max Drawdown: $MAX_DRAWDOWN%"
echo "Mode: $([ -n "$USE_SPOT" ] && echo "Spot" || echo "USDM Futures")"
if [ -z "$USE_SPOT" ]; then
    echo "Leverage: ${LEVERAGE}x"
    echo "Margin Type: $([ -n "$MARGIN_TYPE" ] && echo "Cross" || echo "Isolated")"
fi
echo "-----------------------------------------"
echo "This will execute REAL trades with REAL money."
echo "Press Ctrl+C now to abort if you don't want to proceed."
echo "-----------------------------------------"
echo "Continuing in 5 seconds..."
sleep 5
echo "Starting real trading now!"
echo "========================================="

# Run the trading bot
./build/src/trading_bot \
    --symbol "$SYMBOL" \
    --duration "$DURATION" \
    --balance "$BALANCE" \
    --stop-loss "$STOP_LOSS" \
    --max-drawdown "$MAX_DRAWDOWN" \
    --api-key "$API_KEY" \
    --secret-key "$SECRET_KEY" \
    $USE_SPOT \
    --leverage "$LEVERAGE" \
    $MARGIN_TYPE &

BOT_PID=$!

echo "Trading bot started with PID: $BOT_PID"
echo "Press Ctrl+C to trigger emergency stop (cancel all orders and exit)"
echo "To cancel all orders from another terminal: kill -SIGUSR1 $BOT_PID"

# Wait for the bot to finish
wait $BOT_PID

# Exit with the same status as the trading bot
exit $? 