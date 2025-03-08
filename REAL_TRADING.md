# Real Trading with Binance API

This document explains how to use the trading bot with real trading enabled using the Binance API.

## Prerequisites

1. A Binance account
2. API key and secret key with trading permissions
3. Sufficient funds in your Binance account

## Security Warning

**IMPORTANT**: Your API keys provide direct access to your Binance account and funds. Always keep them secure:

- Never share your API keys with anyone
- Never commit your API keys to version control
- Use environment variables or secure storage for your keys
- Consider using IP restrictions on your API keys
- Only grant the minimum necessary permissions to your API keys

## Getting Started

### 1. Create API Keys

1. Log in to your Binance account
2. Go to "API Management" in your account settings
3. Create a new API key
4. Enable trading permissions (but not withdrawal permissions for security)
5. Set IP restrictions if possible
6. Save your API key and secret key securely

### 2. Build the Trading Bot

Follow the standard build instructions in the main README.md file.

### 3. Run with Real Trading

You can run the trading bot with real trading enabled using the provided script:

```bash
./run_real_trading.sh <api_key> <secret_key> [options]
```

Or directly:

```bash
./build/src/trading_bot \
    --symbol btcusdt \
    --duration 3600 \
    --balance 100.0 \
    --stop-loss 0.5 \
    --max-drawdown 5.0 \
    --api-key <your_api_key> \
    --secret-key <your_secret_key> \
    --real-trading
```

### Options

- `--symbol`: Trading symbol (default: btcusdt)
- `--duration`: Test duration in seconds (default: 3600)
- `--balance`: Initial balance (default: 100.0)
- `--stop-loss`: Stop loss percentage (default: 0.5%)
- `--max-drawdown`: Max drawdown percentage (default: 5.0%)

### USDM Futures Options

By default, the trading bot uses Binance USDM Perpetual Futures. You can customize the futures trading with these options:

- `--spot`: Use spot trading instead of USDM futures
- `--leverage NUM`: Set leverage for futures trading (1-125, default: 5)
- `--cross-margin`: Use cross margin instead of isolated margin (default is isolated margin)

Example for futures trading with 10x leverage:

```bash
./run_real_trading.sh <api_key> <secret_key> --symbol btcusdt --leverage 10
```

Example for spot trading:

```bash
./run_real_trading.sh <api_key> <secret_key> --symbol btcusdt --spot
```

## Stopping and Interfering with Trades

The trading bot provides several ways to stop or interfere with trading:

### Normal Stop

- **Keyboard Interrupt (Ctrl+C)**: Press Ctrl+C to gracefully stop the bot. This will complete the current operation and then exit.
- **Time Limit**: The bot automatically stops after the duration specified with `--duration` (default: 3600 seconds/1 hour).

### Emergency Stop

For immediate intervention:

1. **Emergency Stop (Ctrl+C in script)**: When running through the `run_real_trading.sh` script, pressing Ctrl+C will trigger an emergency stop that cancels all open orders before exiting.

2. **Send SIGUSR1 Signal**: From another terminal, you can send a SIGUSR1 signal to cancel all open orders:
   ```bash
   kill -SIGUSR1 <bot_pid>
   ```
   The PID is displayed when the bot starts.

3. **Risk Management Auto-Stop**: The bot automatically stops trading when:
   - Stop loss is triggered (set with `--stop-loss`)
   - Max drawdown is reached (set with `--max-drawdown`)

### Manual Order Management

If you need to manually manage orders while the bot is running:

1. Use the Binance website or app to view and cancel orders
2. Any changes made on the Binance platform will be detected by the bot on the next update cycle

## Risk Management

The trading bot includes several risk management features:

1. **Stop Loss**: Automatically stops trading if a single trade loses more than the specified percentage
2. **Max Drawdown**: Automatically stops trading if the total drawdown exceeds the specified percentage
3. **Order Size Scaling**: Automatically scales order size based on account balance
4. **Leverage Control**: For futures trading, you can set the leverage to control risk exposure

## Monitoring

While the bot is running, it will display performance metrics including:

- Tick-to-trade latency
- Trades per second
- Success rate
- P&L metrics
- USD volume metrics

## Troubleshooting

If you encounter issues with real trading:

1. **Connection Issues**: Ensure your internet connection is stable
2. **API Key Issues**: Verify your API keys are correct and have the necessary permissions
3. **Insufficient Funds**: Ensure you have sufficient funds in your Binance account
4. **Order Placement Failures**: Check the error messages for specific reasons
5. **Futures-Specific Issues**:
   - Ensure your account has futures trading enabled
   - Check if you have sufficient margin for the selected leverage
   - Verify that the symbol is available for futures trading

## Disclaimer

Trading cryptocurrencies involves significant risk. This trading bot is provided for educational and research purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this software.

**Additional Warning for Futures Trading**: Futures trading with leverage significantly increases risk. You can lose more than your initial investment. Only use leverage that you fully understand and are comfortable with. 