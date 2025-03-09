#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import json
from collections import Counter

# Import the model architecture from train_model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_model import TradingModel

class TradingHistoryDataset(Dataset):
    def __init__(self, features, labels, weights=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.features[idx], self.labels[idx], self.weights[idx]
        return self.features[idx], self.labels[idx]

def process_trading_history(history_path, profit_threshold=0.0):
    """
    Process trading history data for model retraining.
    
    Args:
        history_path: Path to the trading history CSV file
        profit_threshold: Minimum profit to consider a trade successful
        
    Returns:
        features: Numpy array of features
        labels: Numpy array of labels (1 for successful trades, 0 for unsuccessful)
        weights: Sample weights based on profit/loss magnitude
    """
    print(f"Processing trading history from {history_path}")
    
    # Load the trading history
    df = pd.read_csv(history_path)
    
    # Filter out incomplete trades
    df = df[df['Completed'] == 'TRUE']
    
    if len(df) == 0:
        print("No completed trades found in the history")
        return None, None, None
    
    print(f"Found {len(df)} completed trades")
    
    # Extract features
    feature_cols = [col for col in df.columns if col.startswith('Feature')]
    features = df[feature_cols].values
    
    # Create labels based on profit/loss
    # 1 for profitable trades, 0 for unprofitable trades
    labels = (df['ProfitLoss'] > profit_threshold).astype(int).values
    
    # Create sample weights based on the magnitude of profit/loss
    # This gives more importance to trades with larger profits or losses
    weights = np.abs(df['ProfitLoss'].values)
    
    # Normalize weights to have mean=1
    if len(weights) > 0 and np.sum(weights) > 0:
        weights = weights / np.mean(weights)
    else:
        weights = np.ones_like(weights)
    
    # Print class distribution
    class_counts = Counter(labels)
    print(f"Class distribution:")
    print(f"  Profitable trades: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(labels)*100:.2f}%)")
    print(f"  Unprofitable trades: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(labels)*100:.2f}%)")
    
    return features, labels, weights

def retrain_model(features, labels, weights, original_model_path, output_path, 
                 batch_size=32, epochs=20, learning_rate=0.001, val_split=0.2):
    """
    Retrain the model using trading history data.
    
    Args:
        features: Numpy array of features
        labels: Numpy array of labels
        weights: Sample weights
        original_model_path: Path to the original model
        output_path: Path to save the retrained model
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        val_split: Validation split ratio
    """
    # Load the original model
    try:
        original_model = torch.jit.load(original_model_path)
        print(f"Loaded original model from {original_model_path}")
    except Exception as e:
        print(f"Error loading original model: {e}")
        print("Creating a new model instead")
        original_model = None
    
    # Get input size from features
    input_size = features.shape[1]
    print(f"Input size: {input_size}")
    
    # Create a new model with the same architecture
    model = TradingModel(input_size=input_size)
    
    # If we have an original model, copy its weights
    if original_model is not None:
        try:
            # Extract state dict from TorchScript model
            state_dict = {}
            for name, param in original_model.named_parameters():
                state_dict[name] = param.detach()
            
            # Load state dict into the new model
            model.load_state_dict(state_dict)
            print("Transferred weights from original model")
        except Exception as e:
            print(f"Error transferring weights: {e}")
            print("Starting with fresh weights")
    
    # Create dataset
    dataset = TradingHistoryDataset(features, labels, weights)
    
    # Split into training and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # We'll apply weights manually
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                inputs, targets, batch_weights = batch
                batch_weights = batch_weights.view(-1, 1)
            else:
                inputs, targets = batch
                batch_weights = torch.ones_like(targets).view(-1, 1)
            
            targets = targets.view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            weighted_loss = (loss * batch_weights).mean()
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, targets, batch_weights = batch
                    batch_weights = batch_weights.view(-1, 1)
                else:
                    inputs, targets = batch
                    batch_weights = torch.ones_like(targets).view(-1, 1)
                
                targets = targets.view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                weighted_loss = (loss * batch_weights).mean()
                
                val_loss += weighted_loss.item()
                
                # Store predictions and targets for metrics
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_pred_classes = (val_preds > 0.5).astype(int)
        
        accuracy = accuracy_score(val_targets, val_pred_classes)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    # Calculate final metrics
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_pred_classes = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(all_targets, all_pred_classes)
    
    if len(np.unique(all_targets)) > 1:  # Only calculate these if we have both classes
        metrics['precision'] = precision_score(all_targets, all_pred_classes)
        metrics['recall'] = recall_score(all_targets, all_pred_classes)
        metrics['f1_score'] = f1_score(all_targets, all_pred_classes)
        metrics['roc_auc'] = roc_auc_score(all_targets, all_preds)
    
    # Print confusion matrix
    cm = confusion_matrix(all_targets, all_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print metrics
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save the model
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    print(f"Model saved to {output_path}")
    
    # Save metrics
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plot_path = os.path.splitext(output_path)[0] + "_loss.png"
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Retrain trading model using trading history')
    parser.add_argument('--history', type=str, required=True, help='Path to the trading history CSV file')
    parser.add_argument('--original-model', type=str, required=True, help='Path to the original model')
    parser.add_argument('--output', type=str, default='model_retrained.pt', help='Path to save the retrained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--profit-threshold', type=float, default=0.0, 
                        help='Minimum profit to consider a trade successful')
    
    args = parser.parse_args()
    
    # Process trading history
    features, labels, weights = process_trading_history(args.history, args.profit_threshold)
    
    if features is None or len(features) == 0:
        print("No valid trading history data found. Exiting.")
        return
    
    # Retrain the model
    retrain_model(features, labels, weights, args.original_model, args.output,
                 batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate)

if __name__ == "__main__":
    main() 