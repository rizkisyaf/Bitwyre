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

## Workflow

1. Collect orderbook data using the C++ `collect_orderbook_data` executable
2. Create a varied model using `create_varied_model.py`
3. Test the model using the `test_varied_model.sh` script
4. Update the model path in `main.cpp` to use the new model

## Model Architecture

The trading model uses a simple neural network with the following features:
- Input layer with configurable size (default: 25)
- Hidden layer with LayerNorm and GELU activation
- Dropout for regularization
- Output layer with a single neuron
- Sigmoid activation for binary classification

## Training Process

The model is initialized with weights that produce varied predictions:
- Xavier/Glorot initialization with higher gain for 2D weights
- Normal initialization with higher standard deviation for 1D weights
- Uniform initialization for biases to break symmetry

The model is tested with random inputs to ensure it produces varied predictions with good distribution of BUY, SELL, and NEUTRAL signals.

## Prerequisites

Install the required Python packages:

```bash
pip install torch numpy pandas matplotlib scikit-learn
``` 