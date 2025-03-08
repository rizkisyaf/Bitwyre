#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ImprovedTraining")

# Import the model architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.convert_model import ImprovedTradingModel

class OrderbookDataset:
    def __init__(self, csv_file, prediction_horizon=5, feature_window=20):
        """
        Initialize the dataset with orderbook data
        
        Args:
            csv_file: Path to the CSV file with orderbook data
            prediction_horizon: Number of future rows to look ahead for target calculation
            feature_window: Number of past rows to use for feature calculation
        """
        logger.info(f"Loading data from {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        
        # Convert timestamp to datetime for time-based features
        self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='ms')
        
        # Basic preprocessing
        self._preprocess_data()
        
        # Engineer features
        self.features = self._engineer_features()
        
        # Generate targets
        self.targets = self._generate_targets()
        
        # Align features and targets
        self._align_data()
        
        # Split data
        self._split_data()
        
        # Scale features
        self._scale_features()
        
        logger.info(f"Dataset prepared with {len(self.X_train)} training samples and {len(self.X_test)} test samples")
    
    def _preprocess_data(self):
        """Basic preprocessing of the data"""
        # Calculate mid prices
        self.data['mid_price'] = (self.data['bid_price1'] + self.data['ask_price1']) / 2
        
        # Calculate spreads
        self.data['spread'] = self.data['ask_price1'] - self.data['bid_price1']
        
        # Calculate total bid and ask volumes (first 5 levels)
        self.data['bid_volume'] = sum([self.data[f'bid_qty{i}'] for i in range(1, 6)])
        self.data['ask_volume'] = sum([self.data[f'ask_qty{i}'] for i in range(1, 6)])
        
        # Calculate imbalance
        total_volume = self.data['bid_volume'] + self.data['ask_volume']
        self.data['imbalance'] = (self.data['bid_volume'] - self.data['ask_volume']) / total_volume.replace(0, np.nan).fillna(0)
        
        # Calculate price changes
        self.data['price_change'] = self.data['mid_price'].pct_change()
        
        # Fill NaN values
        self.data = self.data.fillna(0)
        
        logger.info("Basic preprocessing completed")
    
    def _engineer_features(self):
        """Engineer features from the orderbook data"""
        logger.info("Engineering features")
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=self.data.index)
        
        # Basic price features
        features['mid_price'] = self.data['mid_price']
        features['spread'] = self.data['spread']
        features['spread_pct'] = self.data['spread'] / self.data['mid_price']
        
        # Volume features
        features['bid_volume'] = self.data['bid_volume']
        features['ask_volume'] = self.data['ask_volume']
        features['volume_ratio'] = self.data['bid_volume'] / self.data['ask_volume'].replace(0, np.nan).fillna(0.001)
        features['imbalance'] = self.data['imbalance']
        
        # Price level features (first 5 levels)
        for i in range(1, 6):
            # Price differences between levels
            if i < 5:
                features[f'bid_price_diff_{i}'] = (self.data[f'bid_price{i}'] - self.data[f'bid_price{i+1}']) / self.data[f'bid_price{i}']
                features[f'ask_price_diff_{i}'] = (self.data[f'ask_price{i+1}'] - self.data[f'ask_price{i}']) / self.data[f'ask_price{i}']
            
            # Volume at each level normalized by total volume
            features[f'bid_vol_norm_{i}'] = self.data[f'bid_qty{i}'] / self.data['bid_volume'].replace(0, np.nan).fillna(0.001)
            features[f'ask_vol_norm_{i}'] = self.data[f'ask_qty{i}'] / self.data['ask_volume'].replace(0, np.nan).fillna(0.001)
        
        # Time-based features
        features['hour_of_day'] = self.data['datetime'].dt.hour
        features['minute_of_hour'] = self.data['datetime'].dt.minute
        features['day_of_week'] = self.data['datetime'].dt.dayofweek
        
        # Rolling window features
        for window in [5, 10, 20]:
            # Price volatility
            features[f'volatility_{window}'] = self.data['mid_price'].pct_change().rolling(window).std().fillna(0)
            
            # Price momentum
            features[f'momentum_{window}'] = self.data['mid_price'].pct_change(window).fillna(0)
            
            # Volume momentum
            features[f'volume_momentum_{window}'] = (self.data['bid_volume'] + self.data['ask_volume']).pct_change(window).fillna(0)
            
            # Imbalance momentum
            features[f'imbalance_momentum_{window}'] = self.data['imbalance'].diff(window).fillna(0)
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Remove outliers (replace with median of column)
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64]:
                median = features[col].median()
                q1, q3 = features[col].quantile(0.25), features[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 3 * iqr, q3 + 3 * iqr
                features.loc[(features[col] < lower_bound) | (features[col] > upper_bound), col] = median
        
        logger.info(f"Created {len(features.columns)} features")
        return features
    
    def _generate_targets(self):
        """Generate target variables based on future price movements"""
        logger.info(f"Generating targets with prediction horizon of {self.prediction_horizon}")
        
        # Initialize targets DataFrame
        targets = pd.DataFrame(index=self.data.index)
        
        # Future returns at different horizons
        future_price = self.data['mid_price'].shift(-self.prediction_horizon)
        current_price = self.data['mid_price']
        
        # Calculate future returns
        targets['future_return'] = (future_price - current_price) / current_price
        
        # Binary classification target (1 if price goes up, 0 if down)
        targets['price_direction'] = (targets['future_return'] > 0).astype(float)
        
        # Multi-class target (0: significant down, 1: slight down, 2: neutral, 3: slight up, 4: significant up)
        returns = targets['future_return']
        thresholds = returns.quantile([0.2, 0.4, 0.6, 0.8])
        
        targets['price_movement'] = 2  # neutral by default
        targets.loc[returns < thresholds[0.2], 'price_movement'] = 0  # significant down
        targets.loc[(returns >= thresholds[0.2]) & (returns < thresholds[0.4]), 'price_movement'] = 1  # slight down
        targets.loc[(returns > thresholds[0.6]) & (returns <= thresholds[0.8]), 'price_movement'] = 3  # slight up
        targets.loc[returns > thresholds[0.8], 'price_movement'] = 4  # significant up
        
        # Drop the last rows that don't have targets
        targets = targets.iloc[:-self.prediction_horizon]
        
        logger.info(f"Generated targets with shape {targets.shape}")
        return targets
    
    def _align_data(self):
        """Align features and targets, handling the feature window and prediction horizon"""
        # Features need to be aligned with targets
        # We need to drop the first feature_window rows (as they don't have enough history)
        # and the last prediction_horizon rows (as they don't have targets)
        
        # Adjust features for the feature window
        self.aligned_features = self.features.iloc[self.feature_window:]
        
        # Adjust targets to match
        self.aligned_targets = self.targets.iloc[:len(self.aligned_features)]
        
        logger.info(f"Aligned data: features shape {self.aligned_features.shape}, targets shape {self.aligned_targets.shape}")
    
    def _split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets"""
        # Convert to numpy arrays
        X = self.aligned_features.values
        y = self.aligned_targets['price_direction'].values  # Using binary classification for now
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        logger.info(f"Data split: X_train shape {self.X_train.shape}, X_test shape {self.X_test.shape}")
    
    def _scale_features(self):
        """Scale the features using StandardScaler"""
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save scaler parameters for later use in the C++ code
        self.feature_mean = self.scaler.mean_
        self.feature_std = self.scaler.scale_
        
        logger.info("Features scaled")
    
    def get_train_loader(self, batch_size=64):
        """Get a DataLoader for the training data"""
        train_dataset = TensorDataset(
            torch.tensor(self.X_train_scaled, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
        )
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_test_loader(self, batch_size=64):
        """Get a DataLoader for the testing data"""
        test_dataset = TensorDataset(
            torch.tensor(self.X_test_scaled, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.float32).unsqueeze(1)
        )
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def save_normalization_params(self, mean_path, std_path):
        """Save the normalization parameters for use in C++"""
        np.save(mean_path, self.feature_mean.astype(np.float32))
        np.save(std_path, self.feature_std.astype(np.float32))
        logger.info(f"Saved normalization parameters to {mean_path} and {std_path}")


def train_model(dataset, model_path, epochs=100, batch_size=64, learning_rate=0.001, patience=10):
    """Train the model and save it"""
    # Get data loaders
    train_loader = dataset.get_train_loader(batch_size)
    test_loader = dataset.get_test_loader(batch_size)
    
    # Initialize model
    input_size = dataset.X_train_scaled.shape[1]
    model = ImprovedTradingModel(input_size=input_size)
    
    # Initialize weights properly
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            nn.init.constant_(param, 0.01)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []
    test_losses = []
    
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Check for improvement
        if test_loss < best_loss:
            best_loss = test_loss
            no_improve_epochs = 0
            
            # Save the best model
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model improved, saved to {model_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    
    logger.info("Training completed")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train an improved trading model')
    parser.add_argument('--data', type=str, required=True, help='Path to the orderbook data CSV file')
    parser.add_argument('--model_output', type=str, default='model_new.pt', help='Path to save the trained model')
    parser.add_argument('--mean_output', type=str, default='mean_new.npy', help='Path to save the feature means')
    parser.add_argument('--std_output', type=str, default='std_new.npy', help='Path to save the feature standard deviations')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--prediction_horizon', type=int, default=5, help='Number of steps to look ahead for prediction')
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = OrderbookDataset(
        args.data, 
        prediction_horizon=args.prediction_horizon
    )
    
    # Train model
    model = train_model(
        dataset, 
        args.model_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save normalization parameters
    dataset.save_normalization_params(args.mean_output, args.std_output)
    
    # Convert model to TorchScript
    model.eval()
    example_input = torch.randn(1, dataset.X_train_scaled.shape[1])
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(args.model_output)
    
    logger.info(f"Traced model saved to {args.model_output}")
    logger.info("Done!")


if __name__ == "__main__":
    main() 