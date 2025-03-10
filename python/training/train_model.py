#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time
import json
from collections import Counter

# Define the model architecture
class TradingModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        # LSTM layers for capturing temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout
        )
        
        # Fully connected layers for prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights properly to avoid NaN issues
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to avoid NaN issues during training"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Use Xavier/Glorot initialization for weight matrices
                    nn.init.xavier_uniform_(param)
                else:
                    # Use uniform initialization for bias terms
                    nn.init.uniform_(param, -0.1, 0.1)
    
    def forward(self, x):
        # Reshape input for LSTM if it's not already in batch_size x seq_len x features format
        batch_size = x.size(0)
        if len(x.shape) == 2:
            # Add sequence dimension (batch_size, 1, features)
            x = x.unsqueeze(1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Apply attention (need to reshape for attention mechanism)
        # Convert to seq_len x batch_size x hidden_size for attention
        attn_input = lstm_out.unsqueeze(0)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        
        # Reshape back to batch_size x hidden_size
        attn_output = attn_output.squeeze(0)
        
        # Pass through fully connected layers
        output = self.fc_layers(attn_output)
        
        return output

# Custom dataset for orderbook data
class OrderbookDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def process_orderbook_data(data_path, future_window=10, price_change_threshold=0.001, neutral_handling='exclude'):
    """
    Process orderbook data from CSV file
    
    Args:
        data_path: Path to the CSV file
        future_window: Number of rows to look ahead for price change
        price_change_threshold: Threshold for price change to be considered significant
        neutral_handling: How to handle neutral labels ('exclude', 'distribute', or 'include')
        
    Returns:
        features: Numpy array of features
        labels: Numpy array of labels
    """
    print(f"Processing orderbook data from {data_path}")
    print(f"Future window: {future_window}, Price change threshold: {price_change_threshold}")
    print(f"Neutral handling: {neutral_handling}")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Initialize lists to store features and labels
    all_features = []
    all_labels = []
    
    # Track mid prices for labeling
    mid_prices = []
    
    # Track additional temporal features
    price_history = []  # For volatility calculation
    volume_history = []  # For volume-based features
    
    # Process each row
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Extract bid and ask data
        bid_prices = []
        bid_quantities = []
        ask_prices = []
        ask_quantities = []
        
        for j in range(1, 11):  # 10 levels
            bid_price_col = f'bid_price_{j}'
            bid_quantity_col = f'bid_quantity_{j}'
            ask_price_col = f'ask_price_{j}'
            ask_quantity_col = f'ask_quantity_{j}'
            
            if bid_price_col in row and not pd.isna(row[bid_price_col]):
                bid_prices.append(float(row[bid_price_col]))
                bid_quantities.append(float(row[bid_quantity_col]))
            
            if ask_price_col in row and not pd.isna(row[ask_price_col]):
                ask_prices.append(float(row[ask_price_col]))
                ask_quantities.append(float(row[ask_quantity_col]))
        
        # Skip if no bids or asks
        if len(bid_prices) == 0 or len(ask_prices) == 0:
            continue
        
        # Calculate mid price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        mid_prices.append(mid_price)
        
        # Calculate spread
        spread = ask_prices[0] - bid_prices[0]
        
        # Calculate bid and ask volumes
        bid_volume = sum(bid_quantities)
        ask_volume = sum(ask_quantities)
        
        # Calculate bid-ask imbalance
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        
        # Update history for temporal features
        price_history.append(mid_price)
        volume_history.append((bid_volume, ask_volume))
        
        # Create feature vector
        row_features = []
        
        # Basic features
        row_features.append(mid_price)
        row_features.append(spread)
        row_features.append(imbalance)
        
        # Add raw prices and quantities
        row_features.extend(bid_prices)
        row_features.extend(bid_quantities)
        row_features.extend(ask_prices)
        row_features.extend(ask_quantities)
        
        # Add price differences between levels
        for j in range(1, len(bid_prices)):
            row_features.append(bid_prices[j-1] - bid_prices[j])
        
        for j in range(1, len(ask_prices)):
            row_features.append(ask_prices[j] - ask_prices[j-1])
            
        # NEW FEATURES
        
        # 1. Depth ratio - ratio of bid volume to ask volume
        depth_ratio = bid_volume / (ask_volume + 1e-10)  # Avoid division by zero
        row_features.append(depth_ratio)
        
        # 2. Momentum features - price changes over different windows
        if i >= 5 and len(price_history) >= 5:
            momentum_5 = mid_price - price_history[-5]
            row_features.append(momentum_5)
        else:
            row_features.append(0.0)
            
        if i >= 20 and len(price_history) >= 20:
            momentum_20 = mid_price - price_history[-20]
            row_features.append(momentum_20)
        else:
            row_features.append(0.0)
        
        # 3. Volatility - standard deviation of price over windows
        if i >= 10 and len(price_history) >= 10:
            volatility_10 = np.std(price_history[-10:])
            row_features.append(volatility_10)
        else:
            row_features.append(0.0)
            
        if i >= 30 and len(price_history) >= 30:
            volatility_30 = np.std(price_history[-30:])
            row_features.append(volatility_30)
        else:
            row_features.append(0.0)
        
        # 4. Volume trend - change in volume over time
        if i >= 10 and len(volume_history) >= 10:
            bid_vol_change = bid_volume - volume_history[-10][0]
            ask_vol_change = ask_volume - volume_history[-10][1]
            row_features.append(bid_vol_change)
            row_features.append(ask_vol_change)
        else:
            row_features.append(0.0)
            row_features.append(0.0)
        
        # 5. Price acceleration - change in momentum
        if i >= 10 and len(price_history) >= 10:
            prev_momentum = price_history[-2] - price_history[-5]
            current_momentum = price_history[-1] - price_history[-4]
            acceleration = current_momentum - prev_momentum
            row_features.append(acceleration)
        else:
            row_features.append(0.0)
        
        # 6. Liquidity imbalance at different levels
        if len(bid_prices) >= 3 and len(ask_prices) >= 3:
            top3_bid_vol = sum(bid_quantities[:3])
            top3_ask_vol = sum(ask_quantities[:3])
            top3_imbalance = (top3_bid_vol - top3_ask_vol) / (top3_bid_vol + top3_ask_vol + 1e-10)
            row_features.append(top3_imbalance)
        else:
            row_features.append(0.0)
            
        # Store features
        all_features.append(row_features)
    
    # Create labels based on future price changes
    for i in range(len(mid_prices) - future_window):
        current_price = mid_prices[i]
        future_price = mid_prices[i + future_window]
        price_change = (future_price - current_price) / current_price
        
        # Determine label based on price change and threshold
        if price_change > price_change_threshold:
            label = 1  # Price goes up
        elif price_change < -price_change_threshold:
            label = 0  # Price goes down
        else:
            # Handle neutral case based on strategy
            if neutral_handling == 'exclude':
                # Skip this sample
                all_features[i] = None
                continue
            elif neutral_handling == 'distribute':
                # Randomly assign to up or down
                label = np.random.choice([0, 1])
            else:  # 'include'
                # Use a third class (2) for neutral
                label = 2
        
        all_labels.append(label)
    
    # Truncate features to match labels
    all_features = all_features[:len(all_labels)]
    
    # Remove None values (excluded neutral samples)
    features_labels = [(f, l) for f, l in zip(all_features, all_labels) if f is not None]
    if not features_labels:
        print("No valid samples after processing")
        return None, None
    
    all_features, all_labels = zip(*features_labels)
    
    # Convert to numpy arrays
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"Processed {len(features)} samples with {len(features[0])} features")
    
    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique, counts):
        label_name = "Up" if label == 1 else "Down" if label == 0 else "Neutral"
        print(f"  {label_name}: {count} ({count/len(labels):.2%})")
    
    return features, labels

def apply_smote(features, labels):
    """
    Apply SMOTE to oversample the minority class
    
    Args:
        features: Numpy array of features
        labels: Numpy array of labels
    
    Returns:
        features_resampled: Resampled features
        labels_resampled: Resampled labels
    """
    print("Applying SMOTE to balance classes...")
    
    # Convert to binary labels if needed
    binary_labels = labels.copy()
    if 0.5 in labels:
        # If we have neutral labels, treat them as a separate class
        pass
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, binary_labels)
    
    # Print class distribution after SMOTE
    counter = Counter(labels_resampled)
    total = len(labels_resampled)
    print("Class distribution after SMOTE:")
    for label, count in counter.items():
        label_name = "Up" if label == 1 else "Down" if label == 0 else "Neutral"
        print(f"  {label_name}: {count} ({count/total:.2%})")
    
    return features_resampled, labels_resampled

def train_model(features, labels, input_size, batch_size=64, epochs=50, learning_rate=0.001, val_split=0.2, 
                use_class_weights=True, use_smote=False):
    """
    Train the trading model
    
    Args:
        features: Numpy array of features
        labels: Numpy array of labels
        input_size: Input size for the model
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        val_split: Validation split ratio
        use_class_weights: Whether to use class weights to handle imbalance
        use_smote: Whether to use SMOTE for oversampling
    
    Returns:
        model: Trained PyTorch model
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Dictionary of evaluation metrics
    """
    # Apply SMOTE if requested
    if use_smote:
        features, labels = apply_smote(features, labels)
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    # Create dataset
    dataset = OrderbookDataset(features_tensor, labels_tensor)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Calculate class weights if requested
    if use_class_weights:
        # Count labels in the training set
        train_labels = [labels[i] for i in train_dataset.indices]
        label_counts = Counter(train_labels)
        
        # Calculate weights (inverse frequency)
        n_samples = len(train_labels)
        class_weights = {label: n_samples / (len(label_counts) * count) for label, count in label_counts.items()}
        
        print("Class weights:")
        for label, weight in class_weights.items():
            label_name = "Up" if label == 1 else "Down" if label == 0 else "Neutral"
            print(f"  {label_name}: {weight:.4f}")
        
        # Create weighted sampler for training data
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders with sampler
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        # Create data loaders without sampler
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TradingModel(input_size=input_size)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15  # Early stopping patience
    epochs_no_improve = 0
    
    print(f"Starting training with {train_size} training samples and {val_size} validation samples...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"  New best model! Val Loss: {val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    # Evaluate the model on validation set
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            outputs = model(batch_features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            val_preds.extend(probs.cpu().numpy())
            val_targets.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    
    # Calculate metrics
    val_preds_binary = (val_preds > 0.5).astype(int)
    accuracy = accuracy_score(val_targets, val_preds_binary)
    precision = precision_score(val_targets, val_preds_binary, zero_division=0)
    recall = recall_score(val_targets, val_preds_binary, zero_division=0)
    f1 = f1_score(val_targets, val_preds_binary, zero_division=0)
    roc_auc = roc_auc_score(val_targets, val_preds)
    conf_matrix = confusion_matrix(val_targets, val_preds_binary).tolist()
    
    # Print metrics
    print("\nValidation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {conf_matrix[0][0]}, FP: {conf_matrix[0][1]}")
    print(f"    FN: {conf_matrix[1][0]}, TP: {conf_matrix[1][1]}")
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'best_val_loss': best_val_loss,
        'confusion_matrix': conf_matrix
    }
    
    return model, train_losses, val_losses, metrics

def save_model(model, output_path, mean_path, std_path, scaler, metrics, input_size=50):
    """
    Save the trained model and normalization parameters
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save the model
        mean_path: Path to save the mean values
        std_path: Path to save the standard deviation values
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
        input_size: Input size of the model
    """
    # Save model in TorchScript format
    model.eval()
    example_input = torch.randn(1, input_size)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_path)
    print(f"Model saved to {output_path}")
    
    # Save normalization parameters
    mean = scaler.mean_.astype(np.float32)
    std = scaler.scale_.astype(np.float32)
    
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"Normalization parameters saved to {mean_path} and {std_path}")
    
    # Save metrics
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

def plot_training_history(train_losses, val_losses, output_path):
    """
    Plot training and validation losses
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Training history plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train a trading model on orderbook data')
    parser.add_argument('--data', type=str, required=True, help='Path to the orderbook data CSV file')
    parser.add_argument('--output', type=str, default='model_trained.pt', help='Path to save the trained model')
    parser.add_argument('--mean', type=str, default='mean.npy', help='Path to save the mean values')
    parser.add_argument('--std', type=str, default='std.npy', help='Path to save the standard deviation values')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--future-window', type=int, default=10, help='Number of rows to look ahead for price movement')
    parser.add_argument('--price-change-threshold', type=float, default=0.001, help='Threshold for significant price movement')
    parser.add_argument('--neutral-handling', type=str, default='exclude', choices=['exclude', 'separate', 'distribute'], 
                        help='How to handle neutral labels: exclude, separate, or distribute')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights to handle imbalance')
    parser.add_argument('--use-smote', action='store_true', help='Use SMOTE to oversample the minority class')
    
    args = parser.parse_args()
    
    # Process data
    features, labels = process_orderbook_data(
        args.data, 
        future_window=args.future_window,
        price_change_threshold=args.price_change_threshold,
        neutral_handling=args.neutral_handling
    )
    
    # Get input size from features
    input_size = features.shape[1]
    print(f"Input size: {input_size}")
    
    # Train model
    model, train_losses, val_losses, metrics = train_model(
        features, 
        labels, 
        input_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_class_weights=args.use_class_weights,
        use_smote=args.use_smote
    )
    
    # Save model and normalization parameters
    save_model(
        model, 
        args.output, 
        args.mean, 
        args.std, 
        None,
        metrics,
        input_size
    )
    
    # Plot training history
    plot_path = os.path.splitext(args.output)[0] + "_training_history.png"
    plot_training_history(train_losses, val_losses, plot_path)
    
    print("\nTraining completed successfully!")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main() 