#!/usr/bin/env python3

import torch
import numpy as np

def test_model(model_path, input_size=61):
    print(f'Testing model: {model_path}')
    model = torch.jit.load(model_path)
    model.eval()
    print('Model loaded successfully')
    
    # Create random input
    input_tensor = torch.randn(1, input_size)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    print(f'Prediction: {prob:.4f}')
    return prob

if __name__ == "__main__":
    test_model('model_trained.pt') 