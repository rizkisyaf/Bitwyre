#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os
import argparse

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
        print(f"Successfully loaded model state dict from {input_path}")
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
    
    # Test forward pass to ensure no NaN outputs
    with torch.no_grad():
        output = model(example_input)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Model produces NaN or Inf outputs. Reinitializing...")
            create_new_model(output_path)
            return
        else:
            print(f"Model test output: {output.item()}")
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"Traced model saved to {output_path}")

def create_new_model(output_path):
    """Create a new model with the correct input size and save it"""
    print("Creating new model with input_size=50")
    model = ImprovedTradingModel(input_size=50)
    
    # Initialize weights properly to avoid NaN outputs
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Use Xavier/Glorot initialization for weights with 2+ dimensions
                nn.init.xavier_uniform_(param)
            else:
                # For 1D weights (e.g., in LayerNorm), use normal initialization
                nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            # Initialize biases to small positive values to avoid dead neurons
            nn.init.constant_(param, 0.01)
    
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 50)
    
    # Test forward pass to ensure no NaN outputs
    with torch.no_grad():
        output = model(example_input)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Model produces NaN or Inf outputs. Reinitializing...")
            return create_new_model(output_path)
        else:
            print(f"Model test output: {output.item()}")
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save(output_path)
    print(f"New model saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert or create a trading model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert an existing model')
    convert_parser.add_argument('--input', type=str, required=True, help='Path to the input model')
    convert_parser.add_argument('--output', type=str, required=True, help='Path to save the output model')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new model')
    create_parser.add_argument('--output', type=str, required=True, help='Path to save the output model')
    
    # Legacy command-line support
    parser.add_argument('input', nargs='?', help='Legacy: Path to the input model')
    parser.add_argument('output', nargs='?', help='Legacy: Path to save the output model')
    
    args = parser.parse_args()
    
    # Handle legacy command-line arguments
    if args.input and args.output and not args.command:
        if args.input == 'new':
            create_new_model(args.output)
        else:
            convert_model(args.input, args.output)
        return
    
    # Handle new command-line arguments
    if args.command == 'convert':
        convert_model(args.input, args.output)
    elif args.command == 'create':
        create_new_model(args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 