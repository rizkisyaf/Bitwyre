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
import time
import json
from collections import Counter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the improved model architecture
class ImprovedTradingModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Bidirectional LSTM to capture patterns in both directions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=4,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights properly
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
        # x shape: [batch_size, seq_len, features]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply attention
        # First transpose to [seq_len, batch_size, hidden_size*2]
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the mean across the sequence dimension
        attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden_size*2]
        attn_mean = attn_out.mean(dim=1)  # [batch_size, hidden_size*2]
        
        # Pass through fully connected layers
        output = self.fc(attn_mean)
        
        return output

# Custom dataset for sequence data
class OrderbookSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def extract_features(row):
    """
    Extract meaningful features from a single orderbook snapshot
    
    Args:
        row: DataFrame row containing orderbook data
        
    Returns:
        List of features
    """
    # Extract bid and ask data
    bid_prices = []
    bid_quantities = []
    ask_prices = []
    ask_quantities = []
    
    for j in range(1, 11):  # 10 levels
        bid_price_col = f'bid_price{j}'
        bid_quantity_col = f'bid_qty{j}'
        ask_price_col = f'ask_price{j}'
        ask_quantity_col = f'ask_qty{j}'
        
        if bid_price_col in row and not pd.isna(row[bid_price_col]):
            bid_prices.append(float(row[bid_price_col]))
            bid_quantities.append(float(row[bid_quantity_col]))
        
        if ask_price_col in row and not pd.isna(row[ask_price_col]):
            ask_prices.append(float(row[ask_price_col]))
            ask_quantities.append(float(row[ask_quantity_col]))
    
    # Skip if no bids or asks
    if len(bid_prices) == 0 or len(ask_prices) == 0:
        return None
    
    # Calculate mid price and spread
    mid_price = (bid_prices[0] + ask_prices[0]) / 2
    spread = ask_prices[0] - bid_prices[0]
    spread_pct = spread / mid_price  # Normalized spread
    
    # Calculate total volumes
    bid_volume = sum(bid_quantities)
    ask_volume = sum(ask_quantities)
    
    # Calculate imbalance
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
    
    # Calculate orderbook shape features
    if len(bid_prices) >= 3:
        bid_curve_steepness = (bid_prices[0] - bid_prices[min(len(bid_prices)-1, 9)]) / (sum(bid_quantities) + 1e-10)
        top3_bid_volume = sum(bid_quantities[:3])
        bid_pressure = top3_bid_volume / (bid_volume + 1e-10)
    else:
        bid_curve_steepness = 0
        bid_pressure = 0
    
    if len(ask_prices) >= 3:
        ask_curve_steepness = (ask_prices[min(len(ask_prices)-1, 9)] - ask_prices[0]) / (sum(ask_quantities) + 1e-10)
        top3_ask_volume = sum(ask_quantities[:3])
        ask_pressure = top3_ask_volume / (ask_volume + 1e-10)
    else:
        ask_curve_steepness = 0
        ask_pressure = 0
    
    # Calculate imbalance at different levels
    level_imbalances = []
    for i in range(min(len(bid_quantities), len(ask_quantities), 5)):  # Up to 5 levels
        level_imb = (bid_quantities[i] - ask_quantities[i]) / (bid_quantities[i] + ask_quantities[i] + 1e-10)
        level_imbalances.append(level_imb)
    
    # Pad level imbalances if needed
    while len(level_imbalances) < 5:
        level_imbalances.append(0)
    
    # Extract trade data if available
    taker_buy_volume = 0.0
    taker_sell_volume = 0.0
    trade_count = 0
    avg_trade_price = 0.0
    
    if 'taker_buy_base_volume' in row and not pd.isna(row['taker_buy_base_volume']):
        taker_buy_volume = float(row['taker_buy_base_volume'])
    
    if 'taker_sell_base_volume' in row and not pd.isna(row['taker_sell_base_volume']):
        taker_sell_volume = float(row['taker_sell_base_volume'])
    
    if 'trade_count' in row and not pd.isna(row['trade_count']):
        trade_count = float(row['trade_count'])
    
    if 'avg_trade_price' in row and not pd.isna(row['avg_trade_price']):
        avg_trade_price = float(row['avg_trade_price'])
    
    # Calculate trade-based features
    trade_imbalance = 0.0
    if taker_buy_volume + taker_sell_volume > 0:
        trade_imbalance = (taker_buy_volume - taker_sell_volume) / (taker_buy_volume + taker_sell_volume)
    
    # Calculate relative trade volume compared to orderbook depth
    relative_buy_volume = taker_buy_volume / (ask_volume + 1e-10)  # Buy orders consume ask side
    relative_sell_volume = taker_sell_volume / (bid_volume + 1e-10)  # Sell orders consume bid side
    
    # Price deviation between trades and orderbook
    price_deviation = 0.0
    if avg_trade_price > 0:
        price_deviation = (avg_trade_price - mid_price) / mid_price
    
    # Combine features
    features = [
        mid_price,
        spread_pct,
        imbalance,
        bid_curve_steepness,
        ask_curve_steepness,
        bid_pressure,
        ask_pressure,
        trade_imbalance,
        relative_buy_volume,
        relative_sell_volume,
        trade_count,
        price_deviation
    ]
    features.extend(level_imbalances)
    
    return features

def process_orderbook_sequences(data_path, sequence_length=30, future_window=10, price_change_threshold=0.001, neutral_handling='exclude'):
    """
    Process orderbook data from CSV file into sequences
    
    Args:
        data_path: Path to the CSV file
        sequence_length: Number of consecutive orderbook states to include in each sequence
        future_window: Number of rows to look ahead for price change
        price_change_threshold: Threshold for price change to be considered significant
        neutral_handling: How to handle neutral labels ('exclude', 'distribute', or 'include')
        
    Returns:
        sequences: List of sequences, each containing a list of feature vectors
        labels: List of labels
    """
    print(f"Processing orderbook data from {data_path}")
    print(f"Sequence length: {sequence_length}, Future window: {future_window}")
    print(f"Price change threshold: {price_change_threshold}, Neutral handling: {neutral_handling}")
    
    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Extract features from each row
    all_features = []
    mid_prices = []
    
    print("Extracting features from each row...")
    for i in range(len(df)):
        if i % 10000 == 0:
            print(f"  Processing row {i}/{len(df)}")
        
        features = extract_features(df.iloc[i])
        if features is not None:
            all_features.append(features)
            mid_prices.append(features[0])  # Mid price is the first feature
    
    print(f"Extracted features from {len(all_features)} valid rows")
    
    # Create sequences
    sequences = []
    labels = []
    
    print(f"Creating sequences with length {sequence_length}...")
    
    up_count = 0
    down_count = 0
    neutral_count = 0
    
    for i in range(len(all_features) - sequence_length - future_window):
        # Extract sequence
        sequence = all_features[i:i+sequence_length]
        
        # Calculate price change
        current_price = mid_prices[i + sequence_length - 1]
        future_price = mid_prices[i + sequence_length + future_window - 1]
        price_change = (future_price - current_price) / current_price
        
        # Determine label based on price change and threshold
        if price_change > price_change_threshold:
            label = 1  # Price goes up
            up_count += 1
        elif price_change < -price_change_threshold:
            label = 0  # Price goes down
            down_count += 1
        else:
            # Handle neutral case based on strategy
            neutral_count += 1
            if neutral_handling == 'exclude':
                # Skip this sample
                continue
            elif neutral_handling == 'distribute':
                # Randomly assign to up or down
                label = np.random.choice([0, 1])
                if label == 1:
                    up_count += 1
                else:
                    down_count += 1
            else:  # 'include'
                # Use a third class (2) for neutral
                label = 2
        
        sequences.append(sequence)
        labels.append(label)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Label distribution - Up: {up_count}, Down: {down_count}, Neutral: {neutral_count}")
    
    if len(sequences) == 0:
        print("No valid sequences after processing")
        return None, None
    
    return sequences, labels

def train_sequence_model(sequences, labels, input_size, sequence_length, batch_size=32, epochs=50, 
                        learning_rate=0.001, val_split=0.2, use_class_weights=True):
    """
    Train the sequence model on orderbook data
    
    Args:
        sequences: List of sequences, each containing a list of feature vectors
        labels: List of labels
        input_size: Number of features in each vector
        sequence_length: Length of each sequence
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        val_split: Validation split ratio
        use_class_weights: Whether to use class weights to handle imbalance
    
    Returns:
        model: Trained PyTorch model
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Dictionary of evaluation metrics
    """
    # Convert to PyTorch tensors
    print("Converting data to PyTorch tensors...")
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    # Create dataset
    dataset = OrderbookSequenceDataset(sequences_tensor, labels_tensor)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training with {train_size} samples, validating with {val_size} samples")
    
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
    print("Initializing model...")
    model = ImprovedTradingModel(input_size=input_size, hidden_size=128, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15  # Early stopping patience
    epochs_no_improve = 0
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Training on {batch_count} batches...")
        
        for batch_idx, (batch_sequences, batch_labels) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{batch_count} ({batch_idx/batch_count*100:.1f}%)")
            
            optimizer.zero_grad()
            outputs = model(batch_sequences)
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
        
        print(f"Validating on {len(val_loader)} batches...")
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                outputs = model(batch_sequences)
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
        for batch_sequences, batch_labels in val_loader:
            outputs = model(batch_sequences)
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

def save_sequence_model(model, output_path, metrics, input_size, sequence_length):
    """
    Save the trained sequence model and metrics
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save the model
        metrics: Dictionary of evaluation metrics
        input_size: Number of features in each vector
        sequence_length: Length of each sequence
    """
    # Save model in TorchScript format
    model.eval()
    example_input = torch.randn(1, sequence_length, input_size)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_path)
    print(f"Model saved to {output_path}")
    
    # Save model metadata
    metadata = {
        'input_size': input_size,
        'sequence_length': sequence_length,
        'model_type': 'ImprovedTradingModel',
        'features': [
            'mid_price',
            'spread_pct',
            'imbalance',
            'bid_curve_steepness',
            'ask_curve_steepness',
            'bid_pressure',
            'ask_pressure',
            'trade_imbalance',
            'relative_buy_volume',
            'relative_sell_volume',
            'trade_count',
            'price_deviation',
            'level1_imbalance',
            'level2_imbalance',
            'level3_imbalance',
            'level4_imbalance',
            'level5_imbalance'
        ]
    }
    
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_path}")
    
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

def visualize_orderbook_sentiment(sequences, labels, output_path, num_samples=5):
    """
    Visualize orderbook sentiment patterns and their relationship to price movements
    
    Args:
        sequences: List of sequences, each containing a list of feature vectors
        labels: List of labels
        output_path: Path to save the visualizations
        num_samples: Number of samples to visualize
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(sequences), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sequence = sequences[idx]
        label = labels[idx]
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
        
        # Extract features from sequence
        mid_prices = [seq[0] for seq in sequence]
        imbalances = [seq[2] for seq in sequence]
        bid_pressure = [seq[5] for seq in sequence]
        ask_pressure = [seq[6] for seq in sequence]
        level1_imbalance = [seq[12] for seq in sequence]  # Adjusted index for level1_imbalance
        
        # Extract trade-related features
        trade_imbalances = [seq[7] for seq in sequence]
        relative_buy_volumes = [seq[8] for seq in sequence]
        relative_sell_volumes = [seq[9] for seq in sequence]
        trade_counts = [seq[10] for seq in sequence]
        price_deviations = [seq[11] for seq in sequence]
        
        # Plot 1: Mid price
        axes[0].plot(mid_prices)
        axes[0].set_title(f'Mid Price (Future Movement: {"Up" if label == 1 else "Down"})')
        axes[0].grid(True)
        
        # Plot 2: Orderbook imbalance
        axes[1].plot(imbalances)
        axes[1].set_title('Orderbook Imbalance')
        axes[1].axhline(y=0, color='r', linestyle='-')
        axes[1].grid(True)
        
        # Plot 3: Bid/Ask pressure
        axes[2].plot(bid_pressure, 'g', label='Bid Pressure')
        axes[2].plot(ask_pressure, 'r', label='Ask Pressure')
        axes[2].set_title('Bid/Ask Pressure')
        axes[2].legend()
        axes[2].grid(True)
        
        # Plot 4: Level 1 imbalance
        axes[3].plot(level1_imbalance)
        axes[3].set_title('Level 1 Imbalance')
        axes[3].axhline(y=0, color='r', linestyle='-')
        axes[3].grid(True)
        
        # Plot 5: Trade imbalance
        axes[4].plot(trade_imbalances, 'purple')
        axes[4].set_title('Trade Imbalance (Taker Buy vs Sell)')
        axes[4].axhline(y=0, color='r', linestyle='-')
        axes[4].grid(True)
        
        # Plot 6: Relative trade volumes
        axes[5].plot(relative_buy_volumes, 'g', label='Relative Buy Volume')
        axes[5].plot(relative_sell_volumes, 'r', label='Relative Sell Volume')
        axes[5].set_title('Relative Trade Volumes')
        axes[5].legend()
        axes[5].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}_sample{i+1}.png")
        plt.close()
    
    print(f"Saved {num_samples} orderbook sentiment visualizations to {output_path}_sample*.png")

def main():
    parser = argparse.ArgumentParser(description='Train an improved trading model on orderbook data')
    parser.add_argument('--data', type=str, required=True, help='Path to the orderbook data CSV file')
    parser.add_argument('--output', type=str, default='improved_model_trained.pt', help='Path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--sequence-length', type=int, default=30, help='Number of consecutive orderbook states in each sequence')
    parser.add_argument('--future-window', type=int, default=10, help='Number of rows to look ahead for price movement')
    parser.add_argument('--price-change-threshold', type=float, default=0.001, help='Threshold for significant price movement')
    parser.add_argument('--neutral-handling', type=str, default='exclude', choices=['exclude', 'distribute', 'include'], 
                        help='How to handle neutral labels: exclude, distribute, or include')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights to handle imbalance')
    parser.add_argument('--visualize', action='store_true', help='Visualize orderbook sentiment patterns')
    
    args = parser.parse_args()
    
    # Process data into sequences
    sequences, labels = process_orderbook_sequences(
        args.data, 
        sequence_length=args.sequence_length,
        future_window=args.future_window,
        price_change_threshold=args.price_change_threshold,
        neutral_handling=args.neutral_handling
    )
    
    if sequences is None or labels is None:
        print("Error: Failed to process data into sequences")
        return
    
    # Determine input size from the first sequence
    input_size = len(sequences[0][0])
    print(f"Input size: {input_size}")
    
    # Train model
    model, train_losses, val_losses, metrics = train_sequence_model(
        sequences, 
        labels, 
        input_size,
        args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_class_weights=args.use_class_weights
    )
    
    # Save model
    save_sequence_model(
        model, 
        args.output, 
        metrics,
        input_size,
        args.sequence_length
    )
    
    # Plot training history
    plot_path = os.path.splitext(args.output)[0] + "_training_history.png"
    plot_training_history(train_losses, val_losses, plot_path)
    
    # Visualize orderbook sentiment if requested
    if args.visualize:
        viz_path = os.path.splitext(args.output)[0] + "_sentiment"
        visualize_orderbook_sentiment(sequences, labels, viz_path)
    
    print("\nTraining completed successfully!")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
