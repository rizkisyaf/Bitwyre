#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os
import sys

class VariedModel(nn.Module):
    def __init__(self, input_size=25, hidden_size=128):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def create_and_save_model(output_path, input_size=25):
    """Create a model with more varied predictions and save it"""
    print(f"Creating varied model with input_size={input_size}")
    model = VariedModel(input_size=input_size)
    
    # Initialize weights with higher variance to produce more varied predictions
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Use Xavier/Glorot initialization with higher gain
                nn.init.xavier_uniform_(param, gain=5.0)
            else:
                # For 1D weights, use normal initialization with higher std
                nn.init.normal_(param, mean=0.0, std=0.5)
        elif 'bias' in name:
            # Initialize biases with non-zero values to break symmetry
            nn.init.uniform_(param, -0.5, 0.5)
    
    model.eval()
    
    # Create example inputs
    example_inputs = [torch.randn(1, input_size) for _ in range(10)]
    
    # Test forward pass to ensure varied outputs
    with torch.no_grad():
        predictions = []
        for example_input in example_inputs:
            output = model(example_input)
            prob = torch.sigmoid(output).item()
            predictions.append(prob)
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        
        print(f"Prediction Statistics:")
        print(f"Mean: {mean_pred:.4f}")
        print(f"Std Dev: {std_pred:.4f}")
        print(f"Min: {min_pred:.4f}")
        print(f"Max: {max_pred:.4f}")
        
        # Check if predictions are varied enough
        if std_pred < 0.1:
            print("Warning: Model predictions still have low variance. Reinitializing...")
            return create_and_save_model(output_path, input_size)
        
        # Check if predictions are too extreme
        if min_pred < 0.01 or max_pred > 0.99:
            print("Warning: Model predictions are too extreme. Reinitializing...")
            return create_and_save_model(output_path, input_size)
        
        print("Model predictions have good variance!")
    
    # Trace the model with a fixed example input
    traced_model = torch.jit.trace(model, torch.randn(1, input_size))
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"New varied model saved to {output_path}")
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_varied_model.py <input_size> <output_path>")
        sys.exit(1)
    
    input_size = int(sys.argv[1])
    output_path = sys.argv[2]
    
    create_and_save_model(output_path, input_size)

if __name__ == "__main__":
    main() 