import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple model for orderbook prediction
class OrderbookModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=128):
        super(OrderbookModel, self).__init__()
        
        # Increase model capacity for faster data processing
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Add residual connections
        self.residual = nn.Linear(input_size, hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Residual connection
        residual = self.residual(x)
        
        # Main path
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)) + residual)  # Add residual
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output between -1 and 1
        
        return x

# Create and save the model
model = OrderbookModel()

# Set the model to evaluation mode to avoid BatchNorm issues
model.eval()

# Create a batch of examples for BatchNorm
example_input = torch.randn(10, 50)  # Batch size of 10 for BatchNorm
with torch.no_grad():
    # Do a forward pass to initialize BatchNorm parameters
    model(example_input)

# Now trace with a single example
traced_script_module = torch.jit.trace(model, torch.randn(1, 50))
traced_script_module.save("model.pt")

print("Model saved to model.pt") 