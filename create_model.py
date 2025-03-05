import torch
import torch.nn as nn

# Define a simple model for orderbook prediction
class OrderbookModel(nn.Module):
    def __init__(self, input_size=45):
        super(OrderbookModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1 for buy/sell signal
        )
    
    def forward(self, x):
        return self.layers(x)

# Create the model
model = OrderbookModel()

# Create a sample input for tracing
sample_input = torch.randn(1, 45)

# Trace the model to make it compatible with LibTorch
traced_model = torch.jit.trace(model, sample_input)

# Save the model
traced_model.save("model.pt")

print("Model saved to model.pt") 