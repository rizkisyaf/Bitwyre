# Training Scripts

This directory contains scripts for training and managing the trading model.

## Scripts

### `convert_model.py`
- **Purpose**: Converts a PyTorch model to a TorchScript model for use in C++
- **Usage**: `python convert_model.py convert --input <input_model> --output <output_model>`

### `create_varied_model.py`
- **Purpose**: Creates a model with varied predictions to avoid the "always predict 0.5" problem
- **Usage**: `python create_varied_model.py <input_size> <output_path>`

### `update_model_integration.py`
- **Purpose**: Updates the model integration with the C++ code
- **Usage**: `python update_model_integration.py`

### `train_model.py`
- **Purpose**: Trains a new model using orderbook data
- **Usage**: `python train_model.py --data <data_file> --output <output_model> [options]`

### `retrain_model.py`
- **Purpose**: Retrains the model using trading history data to improve predictions based on actual trading results
- **Usage**: `python retrain_model.py --history <history_file> --original-model <model_path> --output <output_model> [options]`
- **Options**:
  - `--batch-size`: Batch size for training (default: 32)
  - `--epochs`: Number of epochs to train (default: 20)
  - `--learning-rate`: Learning rate for the optimizer (default: 0.001)
  - `--profit-threshold`: Minimum profit to consider a trade successful (default: 0.0)

## Workflow

1. Collect orderbook data using the C++ `collect_orderbook_data` executable
2. Train a model using `train_model.py`
3. Test the model using the `test_model.py` script
4. Update the model path in `main.cpp` to use the new model
5. Run the trading bot to collect trading history
6. Retrain the model using `retrain_model.py` with the collected trading history
7. Repeat steps 4-6 to continuously improve the model

## Model Architecture

The trading model uses a neural network with the following features:
- Input layer with configurable size (default: 61)
- Hidden layers with LayerNorm and GELU activation
- Dropout for regularization
- Output layer with a single neuron
- Sigmoid activation for binary classification

## Training Process

The model is trained using binary classification:
- Initial training uses price movement data (up/down)
- Retraining uses actual trading results (profitable/unprofitable)

The retraining process uses:
- Sample weighting based on profit/loss magnitude
- Transfer learning from the previous model
- Validation to prevent overfitting
- Comprehensive metrics to evaluate performance

## Prerequisites

Install the required Python packages:

```bash
pip install torch numpy pandas matplotlib scikit-learn
``` 