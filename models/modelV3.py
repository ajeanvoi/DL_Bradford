import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out

class ResNetPointNet(nn.Module):
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, num_classes):
        super(ResNetPointNet, self).__init__()
        self.fc_initial = nn.Linear(input_size, hidden_layer1_size)
        self.bn_initial = nn.BatchNorm1d(hidden_layer1_size)

        self.layer1 = self.make_layer(hidden_layer1_size, hidden_layer1_size, 2)
        self.layer2 = self.make_layer(hidden_layer1_size, hidden_layer2_size, 2, downsample=nn.Sequential(
            nn.Linear(hidden_layer1_size, hidden_layer2_size),
            nn.BatchNorm1d(hidden_layer2_size),
        ))

        self.fc_final = nn.Linear(hidden_layer2_size, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, downsample=None):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn_initial(self.fc_initial(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.fc_final(out)
        return out

# Example usage
if __name__ == "__main__":
    input_size = 13  # Example input size
    hidden_layer1_size = 128
    hidden_layer2_size = 256
    num_classes = 10  # Example number of classes

    model = ResNetPointNet(input_size, hidden_layer1_size, hidden_layer2_size, num_classes)
    print(model)
