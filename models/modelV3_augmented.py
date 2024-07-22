import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        if in_features != out_features:
            self.match_dimensions = nn.Linear(in_features, out_features)
        else:
            self.match_dimensions = None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))

        if self.match_dimensions:
            residual = self.match_dimensions(residual)

        out += residual
        return F.relu(out)

class ResNetPointNet(nn.Module):
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, num_classes):
        super(ResNetPointNet, self).__init__()
        self.num_classes = num_classes
        self.fc_initial = nn.Linear(input_size, hidden_layer1_size)
        self.bn_initial = nn.BatchNorm1d(hidden_layer1_size)
        self.res_block1 = ResNetBlock(hidden_layer1_size, hidden_layer1_size)
        self.res_block2 = ResNetBlock(hidden_layer1_size, hidden_layer2_size)
        self.fc_final = nn.Linear(hidden_layer2_size, num_classes)

    def forward(self, x):
        #print(f'Input shape: {x.shape}')
        x = x.squeeze(1)  # Remove the singleton dimension if it exists
        #print(f'Shape after squeeze: {x.shape}')
        
        # Check if the input size matches the expected input size
        if x.shape[1] != self.fc_initial.in_features:
            raise ValueError(f"Expected input feature size {self.fc_initial.in_features}, but got {x.shape[1]}")
        
        x = F.relu(self.bn_initial(self.fc_initial(x)))
        #print(f'After initial FC and BN: {x.shape}')
        
        x = self.res_block1(x)
        #print(f'After ResNet block 1: {x.shape}')
        
        x = self.res_block2(x)
        #print(f'After ResNet block 2: {x.shape}')
        
        x = self.fc_final(x)
        return x
