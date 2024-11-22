import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/graph_dummy')

class CombinedModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
    
    def forward(self, x):
        output_a = self.model_a(x)
        output_b = self.model_b(output_a)
        return output_b

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.fc(x)

# Create instances of the models
model_a = ModelA()
model_b = ModelB()

# Combine both models
combined_model = CombinedModel(model_a, model_b)

# Create input tensor
x = torch.randn(1, 10, requires_grad=True)

# Pass input through the combined model
output = combined_model(x)

# Compute loss and backpropagate
loss = output.sum()
loss.backward()

# Log the entire computational graph
writer.add_graph(combined_model, (x,))

writer.close()
