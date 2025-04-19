import torch.nn as nn

class ENet(nn.Module):
    def __init__(self, input_dim):
        """
        Define a neural network for predicting e(x).
        Args:
            input_dim (int): Number of input features.
        """
        super(ENet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.fc(x)
