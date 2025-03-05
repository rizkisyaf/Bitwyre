import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os

class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(TradingModel, self).__init__()
        
        # Define layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh())  # Output between -1 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def generate_synthetic_data(num_samples, input_size):
    """Generate synthetic orderbook data for training"""
    X = np.random.randn(num_samples, input_size)
    
    # Generate labels based on some pattern
    # For example, if the first feature is positive and the second is negative, predict buy (1)
    # Otherwise, predict sell (-1)
    y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        if X[i, 0] > 0 and X[i, 1] < 0:
            y[i] = 1.0  # Buy
        elif X[i, 0] < 0 and X[i, 1] > 0:
            y[i] = -1.0  # Sell
        else:
            # Random small signal
            y[i] = np.random.uniform(-0.3, 0.3)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_model(model, X_train, y_train, num_epochs=100, batch_size=32, learning_rate=0.001):
    """Train the model on the provided data"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train and export a PyTorch trading model')
    parser.add_argument('--input_size', type=int, default=23, help='Input size (number of features)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of synthetic samples to generate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--output_path', type=str, default='model.pt', help='Path to save the model')
    
    args = parser.parse_args()
    
    # Create model
    model = TradingModel(args.input_size, args.hidden_size, args.num_layers)
    
    # Generate synthetic data
    X_train, y_train = generate_synthetic_data(args.num_samples, args.input_size)
    
    # Train model
    model = train_model(model, X_train, y_train, args.num_epochs)
    
    # Export model
    model.eval()
    example_input = torch.randn(1, args.input_size)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(args.output_path)
    
    print(f'Model saved to {args.output_path}')

if __name__ == '__main__':
    main()