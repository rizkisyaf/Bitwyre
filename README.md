# Trading Bot with Machine Learning

This repository contains a trading bot that uses machine learning to predict price movements and execute trades on cryptocurrency exchanges.

## Overview

The trading bot consists of several components:
- **Data Collection**: Collects orderbook data from Binance
- **Model Training**: Trains a machine learning model to predict price movements
- **Trading Bot**: Uses the trained model to make trading decisions

## Getting Started

### Prerequisites

- C++17 compatible compiler
- CMake 3.10 or higher
- Python 3.8 or higher
- PyTorch 1.8 or higher
- Binance API key and secret (for live trading)

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```
   pip install torch pandas numpy matplotlib scikit-learn
   ```
3. Build the C++ components:
   ```
   mkdir -p build
   cd build
   cmake ..
   make -j4
   ```

## Usage

### Data Collection

To collect orderbook data for training:

```bash
./build/src/collect_orderbook_data <symbol> <duration_minutes> <output_file>
```

Example:
```bash
./build/src/collect_orderbook_data btcusdt 120 data/orderbook_data_new.csv
```

### Model Training

To train a new model using the collected data:

```bash
./train_and_deploy.sh
```

This script will:
1. Train a new model using the collected data
2. Test the model to ensure it produces meaningful predictions
3. Convert the model for use with the C++ trading bot
4. Copy the necessary files to the correct locations

### Running the Trading Bot

To run the trading bot with the trained model:

```bash
./build/src/trading_bot <symbol> <duration_seconds> <api_key> <api_secret> [options]
```

Example:
```bash
./build/src/trading_bot btcusdt 3600 your_api_key your_api_secret --initial_balance 1000 --stop_loss 0.5 --max_drawdown 5.0
```

For testing without real trades:
```bash
./test_bot.sh
```

## Model Architecture

The trading model uses a neural network with self-attention mechanisms to predict price movements. Key features:

- Multi-head self-attention for capturing complex patterns in orderbook data
- Residual connections and layer normalization for stable training
- Proper weight initialization to avoid NaN outputs
- Binary classification output (price up or down)

## Troubleshooting

### Model Always Predicts the Same Value

If the model consistently predicts values around 0.5 with low confidence:

1. Collect more diverse training data over a longer period
2. Retrain the model using the `train_and_deploy.sh` script
3. Verify the model produces varied predictions using the test script

### NaN Values in Features

The code includes robust handling for NaN values in features, but if you encounter issues:

1. Check the volatility calculation in `TradingBot.cpp`
2. Ensure proper error handling in feature extraction
3. Verify the model initialization in `convert_model.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.