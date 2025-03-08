#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
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
        
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ff(x))

class ImprovedTradingModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
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

def convert_model(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist")
        return
    
    try:
        # Try to load as a state dict first
        state_dict = torch.load(input_path)
        model = ImprovedTradingModel()
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Could not load as state dict: {e}")
        print("Trying to load as TorchScript model...")
        try:
            # Try to load as a TorchScript model
            old_model = torch.jit.load(input_path, map_location=torch.device('cpu'))
            print("Successfully loaded TorchScript model")
            
            # Create a new model with the correct input size
            model = ImprovedTradingModel()
            
            # We can't directly transfer weights due to different input sizes
            # So we'll create a new model from scratch
            print("Creating new model with input_size=50")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    
    model.eval()
    
    # Create example input with correct shape (batch_size, input_size)
    example_input = torch.randn(1, 50)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"Traced model saved to {output_path}")

def create_new_model(output_path):
    """Create a new model with the correct input size and save it"""
    print("Creating new model with input_size=50")
    model = ImprovedTradingModel(input_size=50)
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 50)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"New model saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_model.py <input_path> <output_path>")
        print("   or: python convert_model.py new <output_path>")
        sys.exit(1)
    
    if sys.argv[1] == "new":
        if len(sys.argv) != 3:
            print("Usage: python convert_model.py new <output_path>")
            sys.exit(1)
        output_path = sys.argv[2]
        create_new_model(output_path)
    else:
        if len(sys.argv) != 3:
            print("Usage: python convert_model.py <input_path> <output_path>")
            sys.exit(1)
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        convert_model(input_path, output_path) 