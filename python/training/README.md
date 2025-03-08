# Improved Trading Model Training and Integration

This directory contains scripts for training an improved trading model and integrating it with the C++ codebase.

## Overview

The improved model features:
- Deeper architecture with residual connections
- Dropout for regularization
- Batch normalization for faster training
- Proper feature normalization
- Early stopping and learning rate scheduling
- Trading-specific evaluation metrics

## Files

- `improved_model.py`: Script for training the improved model
- `update_model_integration.py`: Script for updating the C++ code to work with the improved model
- `load_model.py`: Script for loading the latest model and its metadata

## Prerequisites

Install the required Python packages:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## Usage

### 1. Update C++ Code

First, update the C++ code to work with the improved model:

```bash
python update_model_integration.py
```

This will:
- Update `TradingModel.hpp` to include normalization parameters
- Update `TradingModel.cpp` to load and use normalization parameters
- Create a script for loading the latest model

### 2. Train the Model

Train the improved model using synthetic or real data:

```bash
# Train with synthetic data
python improved_model.py --num_samples 10000 --num_epochs 100 --output_dir ../../models

# Train with real data
python improved_model.py --data_path path/to/orderbook_data.csv --num_epochs 100 --output_dir ../../models
```

The script will:
- Generate synthetic data or load real data
- Preprocess the data (normalization, train/val/test split)
- Train the model with early stopping
- Evaluate the model on test data
- Save the model and its metadata

### 3. Load the Model

Load the latest model and copy it to the project directory:

```bash
python load_model.py --target_dir ../../
```

This will:
- Find the latest model in the models directory
- Copy the model and its metadata to the target directory
- Print instructions for using the model in C++

### 4. Rebuild and Run

Rebuild the C++ project and run the trading bot with the improved model:

```bash
cd ../../
rm -rf build && mkdir build && cd build && cmake .. && make
cd ..
./build/src/trading_bot --symbol btcusdt --duration 30 --balance 100.0 --stop-loss 0.5 --max-drawdown 5.0 --api-key YOUR_API_KEY --secret-key YOUR_SECRET_KEY --leverage 5
```

## Data Format

If you want to use real data, it should be in CSV format with the following columns:
- timestamp: Unix timestamp
- bid_price1, bid_qty1, ask_price1, ask_qty1, ...: Orderbook data
- target: Binary label (1 for price up, 0 for price down)

## Customization

You can customize the model architecture and training parameters:

```bash
python improved_model.py --input_size 50 --hidden_size 128 --num_layers 3 --dropout_rate 0.2 --learning_rate 0.001
```

## Collecting Real Data

To collect real orderbook data for training, you can use the Binance API:

```python
import requests
import pandas as pd
import time

def get_orderbook(symbol, limit=10):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    
    # Extract bids and asks
    bids = data['bids']
    asks = data['asks']
    
    # Create a row with timestamp and orderbook data
    row = [int(time.time())]
    
    # Add bids
    for i in range(limit):
        if i < len(bids):
            row.extend([float(bids[i][0]), float(bids[i][1])])
        else:
            row.extend([0.0, 0.0])
    
    # Add asks
    for i in range(limit):
        if i < len(asks):
            row.extend([float(asks[i][0]), float(asks[i][1])])
        else:
            row.extend([0.0, 0.0])
    
    return row

# Collect data
symbol = "BTCUSDT"
data = []

for _ in range(1000):  # Collect 1000 samples
    row = get_orderbook(symbol)
    data.append(row)
    time.sleep(1)  # Wait 1 second

# Create column names
columns = ['timestamp']
for i in range(10):
    columns.extend([f'bid_price{i+1}', f'bid_qty{i+1}'])
for i in range(10):
    columns.extend([f'ask_price{i+1}', f'ask_qty{i+1}'])

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Add target (future price movement)
# For example, if the price 5 minutes later is higher than the current price, target = 1
df['target'] = 0  # Default to 0

# Save to CSV
df.to_csv('orderbook_data.csv', index=False)
```

## Advanced Usage

### Fine-tuning the Model

If you already have a trained model and want to fine-tune it:

```bash
# TODO: Add fine-tuning functionality
```

### Hyperparameter Optimization

To find the best hyperparameters:

```bash
# TODO: Add hyperparameter optimization functionality
```

## Troubleshooting

- **NaN values in features**: The script automatically replaces NaN values with 0.
- **Model not loading**: Make sure the model file exists and is in the correct format.
- **Compilation errors**: Make sure the C++ code is updated to work with the improved model.
- **Runtime errors**: Check the logs for details on what went wrong.

## Contributing

Feel free to contribute to this project by:
- Adding new features to the model
- Improving the training process
- Enhancing the C++ integration
- Adding more evaluation metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details. 