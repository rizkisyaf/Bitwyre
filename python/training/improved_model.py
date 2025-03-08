#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Trading Model Training Script
--------------------------------------
This script implements an advanced neural network model for predicting market movements
based on orderbook data. It features sophisticated feature engineering, advanced model
architecture, and robust training methodology.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import _LRScheduler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an improved trading model')
    parser.add_argument('--data_path', type=str, help='Path to orderbook data CSV file')
    parser.add_argument('--output_dir', type=str, default='../../models', help='Directory to save model and plots')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    
    return parser.parse_args()

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    time_series_splits: int = 5
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    dropout_rate: float = 0.2

class OrderbookDataset(Dataset):
    def __init__(self, raw_data: pd.DataFrame, scaler: Optional[RobustScaler] = None, is_training: bool = True):
        """Initialize dataset with raw orderbook data"""
        self.raw_data = raw_data
        self.is_training = is_training
        self.scaler = scaler
        
        # Generate targets first (this will determine which rows to keep)
        self.targets = self._generate_targets()
        
        # Engineer features (using only the rows that have valid targets)
        self.features = self._engineer_features()
        
        # Convert features to tensor and normalize
        if self.is_training:
            if self.scaler is None:
                self.scaler = RobustScaler()
                self.features = torch.FloatTensor(
                    self.scaler.fit_transform(self.features)
                )
            else:
                self.features = torch.FloatTensor(
                    self.scaler.transform(self.features)
                )
        else:
            if self.scaler is None:
                raise ValueError("Scaler must be provided for validation/test data")
            self.features = torch.FloatTensor(
                self.scaler.transform(self.features)
            )
        
        # Convert targets to tensor and reshape to match model output
        self.targets = torch.FloatTensor(self.targets).reshape(-1, 1)
    
    def _engineer_features(self):
        df = self.raw_data.copy()
        
        # Drop the last 5 rows since we can't calculate their targets
        df = df.iloc[:-5]
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Calculate price and volume features
        for i in range(1, 11):
            # Price differences
            df[f'bid_ask_spread_{i}'] = df[f'ask_price{i}'] - df[f'bid_price{i}']
            
            # Volume imbalances
            total_volume = df[f'bid_qty{i}'] + df[f'ask_qty{i}']
            df[f'volume_imbalance_{i}'] = (df[f'bid_qty{i}'] - df[f'ask_qty{i}']) / (total_volume + 1e-8)
            
            # Price gaps
            if i < 10:
                df[f'bid_gap_{i}'] = df[f'bid_price{i}'] - df[f'bid_price{i+1}']
                df[f'ask_gap_{i}'] = df[f'ask_price{i+1}'] - df[f'ask_price{i}']
        
        # Calculate cumulative volumes
        bid_volumes = [f'bid_qty{i}' for i in range(1, 11)]
        ask_volumes = [f'ask_qty{i}' for i in range(1, 11)]
        
        df['cum_bid_volume'] = df[bid_volumes].sum(axis=1)
        df['cum_ask_volume'] = df[ask_volumes].sum(axis=1)
        df['total_volume_imbalance'] = (df['cum_bid_volume'] - df['cum_ask_volume']) / (df['cum_bid_volume'] + df['cum_ask_volume'] + 1e-8)
        
        # Select features for the model
        feature_columns = (
            [f'bid_ask_spread_{i}' for i in range(1, 11)] +
            [f'volume_imbalance_{i}' for i in range(1, 11)] +
            [f'bid_gap_{i}' for i in range(1, 10)] +
            [f'ask_gap_{i}' for i in range(1, 10)] +
            ['cum_bid_volume', 'cum_ask_volume', 'total_volume_imbalance', 'hour', 'minute']
        )
        
        return df[feature_columns].values
    
    def _generate_targets(self):
        df = self.raw_data.copy()
        
        # Calculate mid prices
        df['mid_price'] = (df['ask_price1'] + df['bid_price1']) / 2
        
        # Calculate future returns (5 steps ahead)
        df['future_price'] = df['mid_price'].shift(-5)
        df['return'] = (df['future_price'] - df['mid_price']) / df['mid_price']
        
        # Generate target (1 for price increase, 0 for decrease)
        df['target'] = (df['return'] > 0).astype(float)
        
        # Drop last 5 rows since we can't calculate their targets
        df = df.iloc[:-5]
        
        return df['target'].values
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ff(x))

class ImprovedTradingModel(nn.Module):
    """
    Advanced neural network model for trading predictions.
    Features:
    - Multi-head self-attention mechanism
    - Residual connections
    - Layer normalization
    - Advanced activation functions
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 num_heads: int = 4, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Output layers with skip connection
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Add positional dimension for attention
        x = x.unsqueeze(0)
        
        # Apply attention layers
        for layer in self.attention_layers:
            x = layer(x)
        
        # Remove positional dimension
        x = x.squeeze(0)
        
        # Output layers
        return self.output_layers(x)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def generate_synthetic_orderbook_data(num_samples, input_size=50, noise_level=0.1):
    """
    Generate synthetic orderbook data for training.
    
    Parameters:
    - num_samples: Number of samples to generate
    - input_size: Number of features per sample
    - noise_level: Amount of noise to add
    
    Returns:
    - X: Features (orderbook data)
    - y: Labels (price movement direction)
    """
    # Generate base features
    X = np.random.randn(num_samples, input_size) * noise_level
    
    # First 10 features represent bid prices and quantities
    bid_prices = np.sort(np.random.uniform(9000, 10000, (num_samples, 5)))[:, ::-1]
    bid_quantities = np.random.exponential(2, (num_samples, 5))
    
    # Next 10 features represent ask prices and quantities
    ask_prices = np.sort(np.random.uniform(10001, 11000, (num_samples, 5)))
    ask_quantities = np.random.exponential(2, (num_samples, 5))
    
    # Combine prices and quantities
    for i in range(5):
        X[:, i*2] = bid_prices[:, i]
        X[:, i*2+1] = bid_quantities[:, i]
        X[:, 10+i*2] = ask_prices[:, i]
        X[:, 10+i*2+1] = ask_quantities[:, i]
    
    # Calculate mid price
    mid_prices = (bid_prices[:, 0] + ask_prices[:, 0]) / 2
    X[:, 20] = mid_prices
    
    # Calculate spread
    spreads = ask_prices[:, 0] - bid_prices[:, 0]
    X[:, 21] = spreads
    
    # Calculate bid-ask imbalance
    total_bid_qty = np.sum(bid_quantities, axis=1)
    total_ask_qty = np.sum(ask_quantities, axis=1)
    imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
    X[:, 22] = imbalance
    
    # Add some temporal features
    X[:, 23] = np.random.uniform(0, 1, num_samples)  # Time of day
    X[:, 24] = np.random.uniform(0, 1, num_samples)  # Day of week
    
    # Generate labels based on a realistic model
    # If bid volume > ask volume and spread is tight, price likely to go up
    y = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        # Base signal from imbalance
        signal = imbalance[i]
        
        # Adjust based on spread (tighter spread = stronger signal)
        spread_factor = 1.0 / (1.0 + spreads[i] * 0.1)
        
        # Add some randomness
        noise = np.random.normal(0, noise_level)
        
        # Combine factors
        final_signal = signal * spread_factor + noise
        
        # Convert to binary label with some threshold
        y[i] = 1.0 if final_signal > 0.1 else 0.0
    
    return X, y

def load_real_orderbook_data(file_path):
    """
    Load real orderbook data from CSV file.
    Expected format: timestamp, bid_price1, bid_qty1, ask_price1, ask_qty1, ...
    
    Returns:
    - X: Features
    - y: Labels (can be derived from future price movements)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Extract features (all columns except timestamp and target)
        X = df.iloc[:, 1:-1].values
        
        # Extract target (last column)
        y = df.iloc[:, -1].values.reshape(-1, 1)
        
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Falling back to synthetic data")
        return None, None

def preprocess_data(X, y, test_size=0.2, validation_size=0.1):
    """
    Preprocess the data for training:
    - Split into train/validation/test sets
    - Normalize features
    - Convert to PyTorch tensors
    
    Returns:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - test_loader: DataLoader for test data
    - scaler: Fitted StandardScaler for feature normalization
    """
    # Split data into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # Time-based split
    )
    
    # Split train+val into train and validation
    val_ratio = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, shuffle=False  # Time-based split
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: TrainingConfig
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the model and return training history"""
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_metrics = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, config.gradient_clip
        )
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_metrics = _validate_epoch(
                model, val_loader, criterion, device
            )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update history
        for metric in ['loss', 'acc', 'f1']:
            history[f'train_{metric}'].append(train_metrics[metric])
            history[f'val_{metric}'].append(val_metrics[metric])
        
        # Log progress
        logging.info(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model, history

def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    gradient_clip: float
) -> Dict[str, float]:
    """Train for one epoch"""
    total_loss = 0
    all_targets = []
    all_predictions = []
    
    for inputs, batch_targets in train_loader:
        inputs = inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, batch_targets)
        
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        all_targets.extend(batch_targets.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    
    return {'loss': avg_loss, 'acc': accuracy, 'f1': f1}

def _validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate for one epoch"""
    total_loss = 0
    all_targets = []
    all_predictions = []
    
    for inputs, batch_targets in val_loader:
        inputs = inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, batch_targets)
        
        total_loss += loss.item()
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        all_targets.extend(batch_targets.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    
    return {'loss': avg_loss, 'acc': accuracy, 'f1': f1}

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on test set"""
    model.eval()
    all_targets = []
    all_predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, batch_targets in test_loader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(inputs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_targets.extend(batch_targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot detailed training history with multiple metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def save_model_with_metadata(model: nn.Module,
                           scaler: RobustScaler,
                           input_size: int,
                           metrics: Dict[str, float],
                           output_dir: str) -> str:
    """
    Save the model and its metadata for use in C++.
    
    Parameters:
    - model: Trained PyTorch model
    - scaler: Fitted RobustScaler
    - input_size: Number of input features
    - metrics: Dictionary of evaluation metrics
    - output_dir: Directory to save the model
    
    Returns:
    - model_path: Path to the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model
    model.eval()
    example_input = torch.randn(1, input_size)
    traced_model = torch.jit.trace(model, example_input)
    
    model_path = os.path.join(output_dir, f"trading_model_{timestamp}.pt")
    traced_model.save(model_path)
    
    # Save scaler parameters
    scaler_mean_path = os.path.join(output_dir, f"scaler_mean_{timestamp}.npy")
    scaler_std_path = os.path.join(output_dir, f"scaler_std_{timestamp}.npy")
    
    np.save(scaler_mean_path, scaler.center_)
    np.save(scaler_std_path, scaler.scale_)
    
    # Save metadata
    metadata = {
        "model_path": model_path,
        "scaler_mean_path": scaler_mean_path,
        "scaler_std_path": scaler_std_path,
        "input_size": input_size,
        "timestamp": timestamp,
        "metrics": metrics,
        "description": "Advanced trading model with attention mechanism"
    }
    
    metadata_path = os.path.join(output_dir, f"model_metadata_{timestamp}.json")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler parameters saved to {scaler_mean_path} and {scaler_std_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Create a symlink to the latest model
    latest_model_path = os.path.join(output_dir, "model.pt")
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    os.symlink(model_path, latest_model_path)
    
    logger.info(f"Symlink created at {latest_model_path}")
    
    return model_path

def split_data(data: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    # Calculate split indices
    n = len(data)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split data
    train_data = data[:val_idx]
    val_data = data[val_idx:test_idx]
    test_data = data[test_idx:]
    
    logging.info(f"Data split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data

def create_data_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader, RobustScaler]:
    """Create data loaders for training, validation, and test sets."""
    # Create datasets
    train_dataset = OrderbookDataset(train_data, scaler=None, is_training=True)
    val_dataset = OrderbookDataset(val_data, scaler=train_dataset.scaler, is_training=False)
    test_dataset = OrderbookDataset(test_data, scaler=train_dataset.scaler, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.scaler

def main():
    """Main training function"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    logging.info(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    logging.info(f"Loaded data with shape: {data.shape}")
    
    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(data)
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    input_size = len(train_loader.dataset.features[0])
    model = ImprovedTradingModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Create training config
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        patience=args.patience
    )
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=FocalLoss(alpha=0.25, gamma=2.0),
        optimizer=optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        ),
        scheduler=optim.lr_scheduler.OneCycleLR(
            optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            ),
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        ),
        device=device,
        config=config
    )
    
    # Save model and scaler
    model_path = os.path.join(args.output_dir, 'model.pth')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    torch.save(trained_model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate on test set
    metrics = evaluate_model(trained_model, test_loader, device)
    logging.info("\nTest Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save test metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main() 