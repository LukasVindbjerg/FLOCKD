import torch
import torch.nn as nn

# Defining model architectures

# Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
    

# myMLP = SimpleMLP(10, 20, 1)
# print(myMLP)