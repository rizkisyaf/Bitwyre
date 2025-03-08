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
        
        # Input layer
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

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
    Process orderbook data and generate features and labels
    
    Args:
        data_path: Path to the CSV file containing orderbook data
        future_window: Number of rows to look ahead for price movement
        price_change_threshold: Threshold for significant price movement (as a percentage)
        neutral_handling: How to handle neutral labels ('exclude', 'separate', or 'distribute')
    
    Returns:
        features: Numpy array of features
        labels: Numpy array of labels (1 for price up, 0 for price down)
        scaler: Fitted StandardScaler for feature normalization
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if the data has the expected columns
    expected_columns = ['timestamp', 'bid_price1', 'bid_qty1', 'ask_price1', 'ask_qty1']
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract features
    print("Extracting features...")
    
    # Basic features
    features = []
    
    # For each row, extract orderbook features
    for i in range(len(df)):
        row_features = []
        
        # Mid price
        mid_price = (df.loc[i, 'bid_price1'] + df.loc[i, 'ask_price1']) / 2
        row_features.append(mid_price)
        
        # Spread
        spread = df.loc[i, 'ask_price1'] - df.loc[i, 'bid_price1']
        row_features.append(spread)
        
        # Bid-ask imbalance
        bid_volume = 0
        ask_volume = 0
        
        # Sum up bid and ask volumes for all levels
        for j in range(1, 11):  # Assuming 10 levels
            bid_qty_col = f'bid_qty{j}'
            ask_qty_col = f'ask_qty{j}'
            if bid_qty_col in df.columns and ask_qty_col in df.columns:
                bid_volume += df.loc[i, bid_qty_col]
                ask_volume += df.loc[i, ask_qty_col]
        
        # Calculate imbalance
        if (bid_volume + ask_volume) > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            imbalance = 0
        
        row_features.append(imbalance)
        
        # Add all bid and ask prices and quantities as features
        for j in range(1, 11):  # Assuming 10 levels
            bid_price_col = f'bid_price{j}'
            bid_qty_col = f'bid_qty{j}'
            ask_price_col = f'ask_price{j}'
            ask_qty_col = f'ask_qty{j}'
            
            if bid_price_col in df.columns:
                row_features.append(df.loc[i, bid_price_col])
            if bid_qty_col in df.columns:
                row_features.append(df.loc[i, bid_qty_col])
            if ask_price_col in df.columns:
                row_features.append(df.loc[i, ask_price_col])
            if ask_qty_col in df.columns:
                row_features.append(df.loc[i, ask_qty_col])
        
        # Add price differences between levels
        for j in range(1, 10):  # Differences between adjacent levels
            bid_price_col1 = f'bid_price{j}'
            bid_price_col2 = f'bid_price{j+1}'
            ask_price_col1 = f'ask_price{j}'
            ask_price_col2 = f'ask_price{j+1}'
            
            if bid_price_col1 in df.columns and bid_price_col2 in df.columns:
                row_features.append(df.loc[i, bid_price_col1] - df.loc[i, bid_price_col2])
            if ask_price_col2 in df.columns and ask_price_col1 in df.columns:
                row_features.append(df.loc[i, ask_price_col2] - df.loc[i, ask_price_col1])
        
        features.append(row_features)
    
    features = np.array(features, dtype=np.float32)
    print(f"Features extracted. Shape: {features.shape}")
    
    # Handle NaN values
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in features. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Generate labels based on future price movement
    print(f"Generating labels with future_window={future_window} and threshold={price_change_threshold}...")
    labels = []
    
    # Calculate mid prices for all rows
    mid_prices = [(df.loc[i, 'bid_price1'] + df.loc[i, 'ask_price1']) / 2 for i in range(len(df))]
    
    for i in range(len(df) - future_window):
        current_price = mid_prices[i]
        future_price = mid_prices[i + future_window]
        
        # Calculate percentage change
        price_change = (future_price - current_price) / current_price
        
        # 1 if price goes up by threshold, 0 if price goes down by threshold, 0.5 if neutral
        if price_change > price_change_threshold:
            labels.append(1)  # Price goes up
        elif price_change < -price_change_threshold:
            labels.append(0)  # Price goes down
        else:
            if neutral_handling == 'separate':
                labels.append(0.5)  # Neutral as a separate class
            elif neutral_handling == 'distribute':
                # Randomly assign neutral to up or down to maintain balance
                if np.random.random() < 0.5:
                    labels.append(1)
                else:
                    labels.append(0)
            # If 'exclude', we'll filter these out later
    
    # Convert labels to numpy array
    labels = np.array(labels, dtype=np.float32)
    
    # Print initial label distribution
    up_count = np.sum(labels == 1)
    down_count = np.sum(labels == 0)
    neutral_count = np.sum(labels == 0.5)
    total_count = len(labels)
    
    print(f"Initial label distribution:")
    print(f"  Up: {up_count} ({up_count/total_count:.2%})")
    print(f"  Down: {down_count} ({down_count/total_count:.2%})")
    if neutral_handling == 'separate':
        print(f"  Neutral: {neutral_count} ({neutral_count/total_count:.2%})")
    
    # Handle neutral labels based on the specified strategy
    if neutral_handling == 'exclude':
        # Remove neutral samples (those that didn't get labeled as up or down)
        non_neutral_indices = np.where((labels == 1) | (labels == 0))[0]
        features = features[non_neutral_indices]
        labels = labels[non_neutral_indices]
        
        # Remove the last 'future_window' rows from features since we don't have labels for them
        features = features[:-future_window]
    else:
        # Remove the last 'future_window' rows from features since we don't have labels for them
        features = features[:-future_window]
    
    # Print final label distribution
    up_count = np.sum(labels == 1)
    down_count = np.sum(labels == 0)
    neutral_count = np.sum(labels == 0.5)
    total_count = len(labels)
    
    print(f"Final label distribution:")
    print(f"  Up: {up_count} ({up_count/total_count:.2%})")
    print(f"  Down: {down_count} ({down_count/total_count:.2%})")
    if neutral_handling == 'separate':
        print(f"  Neutral: {neutral_count} ({neutral_count/total_count:.2%})")
    
    return features, labels, scaler

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
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
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
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            outputs = model(batch_features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate ROC AUC if possible
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.5  # Default value if calculation fails
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'best_val_loss': float(best_val_loss),
        'confusion_matrix': cm.tolist() if cm is not None else None
    }
    
    print("\nModel Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    if cm is not None and cm.shape == (2, 2):
        print("\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives: {cm[1, 1]}")
    
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
    features, labels, scaler = process_orderbook_data(
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
        scaler,
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