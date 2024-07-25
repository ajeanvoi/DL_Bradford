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

class DeeperResNetPointNet(nn.Module):
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, num_classes, num_res_blocks=4):
        super(DeeperResNetPointNet, self).__init__()
        self.num_classes = num_classes
        self.fc_initial = nn.Linear(input_size, hidden_layer1_size)
        self.bn_initial = nn.BatchNorm1d(hidden_layer1_size)
        
        self.res_blocks = nn.ModuleList([ResNetBlock(hidden_layer1_size, hidden_layer1_size) for _ in range(num_res_blocks - 1)])
        self.res_blocks.append(ResNetBlock(hidden_layer1_size, hidden_layer2_size))
        self.fc_final = nn.Linear(hidden_layer2_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = F.relu(self.bn_initial(self.fc_initial(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.fc_final(x)
        return x

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            _, preds = torch.max(outputs, 1)
        return preds

