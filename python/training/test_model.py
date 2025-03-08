#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ModelTest")

def load_model(model_path):
    """Load a trained TorchScript model"""
    try:
        model = torch.jit.load(model_path)
        model.eval()
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def load_test_data(data_path, mean_path, std_path, test_size=100):
    """Load and preprocess test data"""
    try:
        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} with shape {data.shape}")
        
        # Load normalization parameters
        feature_mean = np.load(mean_path)
        feature_std = np.load(std_path)
        logger.info(f"Loaded normalization parameters from {mean_path} and {std_path}")
        
        # Get the last test_size rows for testing
        test_data = data.tail(test_size)
        
        # Preprocess data (simplified version of what's in improved_training.py)
        # Calculate mid prices
        test_data['mid_price'] = (test_data['bid_price1'] + test_data['ask_price1']) / 2
        
        # Calculate spreads
        test_data['spread'] = test_data['ask_price1'] - test_data['bid_price1']
        
        # Calculate total bid and ask volumes (first 5 levels)
        test_data['bid_volume'] = sum([test_data[f'bid_qty{i}'] for i in range(1, 6)])
        test_data['ask_volume'] = sum([test_data[f'ask_qty{i}'] for i in range(1, 6)])
        
        # Calculate imbalance
        total_volume = test_data['bid_volume'] + test_data['ask_volume']
        test_data['imbalance'] = (test_data['bid_volume'] - test_data['ask_volume']) / total_volume.replace(0, np.nan).fillna(0)
        
        # Extract features (simplified version)
        features = pd.DataFrame(index=test_data.index)
        
        # Basic price features
        features['mid_price'] = test_data['mid_price']
        features['spread'] = test_data['spread']
        features['spread_pct'] = test_data['spread'] / test_data['mid_price']
        
        # Volume features
        features['bid_volume'] = test_data['bid_volume']
        features['ask_volume'] = test_data['ask_volume']
        features['volume_ratio'] = test_data['bid_volume'] / test_data['ask_volume'].replace(0, np.nan).fillna(0.001)
        features['imbalance'] = test_data['imbalance']
        
        # Price level features (first 5 levels)
        for i in range(1, 6):
            # Price differences between levels
            if i < 5:
                features[f'bid_price_diff_{i}'] = (test_data[f'bid_price{i}'] - test_data[f'bid_price{i+1}']) / test_data[f'bid_price{i}']
                features[f'ask_price_diff_{i}'] = (test_data[f'ask_price{i+1}'] - test_data[f'ask_price{i}']) / test_data[f'ask_price{i}']
            
            # Volume at each level normalized by total volume
            features[f'bid_vol_norm_{i}'] = test_data[f'bid_qty{i}'] / test_data['bid_volume'].replace(0, np.nan).fillna(0.001)
            features[f'ask_vol_norm_{i}'] = test_data[f'ask_qty{i}'] / test_data['ask_volume'].replace(0, np.nan).fillna(0.001)
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Convert to numpy array
        X = features.values
        
        # Normalize features
        X_normalized = (X - feature_mean[:X.shape[1]]) / feature_std[:X.shape[1]]
        
        logger.info(f"Prepared test data with shape {X_normalized.shape}")
        return X_normalized, test_data['mid_price'].values
    
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return None, None

def test_model(model, X_test, prices):
    """Test the model on the test data and visualize predictions"""
    try:
        # Convert to tensor
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = []
            for i in range(len(X_tensor)):
                input_tensor = X_tensor[i:i+1]  # Add batch dimension
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
        
        predictions = np.array(predictions)
        logger.info(f"Made predictions with shape {predictions.shape}")
        
        # Check for constant predictions
        unique_preds = np.unique(predictions)
        if len(unique_preds) < 5:
            logger.warning(f"Model predictions have low variance! Unique values: {unique_preds}")
        
        # Calculate prediction statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        logger.info(f"Prediction stats: Mean={mean_pred:.4f}, Std={std_pred:.4f}, Min={min_pred:.4f}, Max={max_pred:.4f}")
        
        # Plot predictions vs prices
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Prices
        plt.subplot(2, 1, 1)
        plt.plot(prices, label='Mid Price')
        plt.title('Price Movement')
        plt.legend()
        
        # Plot 2: Predictions
        plt.subplot(2, 1, 2)
        plt.plot(predictions, label='Model Predictions', color='orange')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.title('Model Predictions (>0.5 = Up, <0.5 = Down)')
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_predictions.png')
        logger.info("Saved prediction visualization to model_predictions.png")
        
        # Check if predictions are meaningful
        if std_pred < 0.01:
            logger.error("Model predictions have very low variance - model may be stuck!")
            return False
        
        if abs(mean_pred - 0.5) < 0.01 and std_pred < 0.05:
            logger.error("Model predictions are centered around 0.5 with low variance - model may not be learning!")
            return False
        
        logger.info("Model predictions appear to be meaningful")
        return True
    
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test a trained trading model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to the test data')
    parser.add_argument('--mean', type=str, required=True, help='Path to the feature means')
    parser.add_argument('--std', type=str, required=True, help='Path to the feature standard deviations')
    parser.add_argument('--test_size', type=int, default=100, help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Load test data
    X_test, prices = load_test_data(args.data, args.mean, args.std, args.test_size)
    if X_test is None:
        return
    
    # Test model
    success = test_model(model, X_test, prices)
    
    if success:
        logger.info("Model test completed successfully")
    else:
        logger.error("Model test failed - model may need retraining")

if __name__ == "__main__":
    main() 