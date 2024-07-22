import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedPointNet(nn.Module):
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, num_classes):
        super(ImprovedPointNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer1_size)
        self.bn1 = nn.BatchNorm1d(hidden_layer1_size)
        self.fc2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.bn2 = nn.BatchNorm1d(hidden_layer2_size)
        self.fc3 = nn.Linear(hidden_layer2_size, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Example usage
if __name__ == "__main__":
    input_size = 3  # Example input size (e.g., x, y, z coordinates)
    hidden_layer1_size = 128
    hidden_layer2_size = 64
    num_classes = 10  # Example number of classes

    model = ImprovedPointNet(input_size, hidden_layer1_size, hidden_layer2_size, num_classes)
    print(model)
