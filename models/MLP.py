import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, in_channels, num_electrodes, hidden_dim, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Define the MLP layers
        self.fc1 = nn.Linear(in_channels * num_electrodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # self.fc3 = nn.Linear(hidden_dim, num_classes)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)  # [batch_size, in_channels * num_electrodes]

        # Pass through MLP layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))

        # Final classification layer
        x = self.fc2(x)

        return x