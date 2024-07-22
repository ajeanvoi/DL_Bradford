import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, num_classes):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer1_size)
        self.fc2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc3 = nn.Linear(hidden_layer2_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
