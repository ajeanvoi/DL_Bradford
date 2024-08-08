import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml

class ResNetBlock(nn.Module):
    ### ResNet block for PointNet
    def __init__(self, in_features, out_features):
        repo_path = '/content/DL_Bradford'
        with open(os.path.join(repo_path, 'configs/model_config.yaml'), 'r') as file:
            model_config = yaml.safe_load(file)

        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(model_config['DeeperResNetPointNet']['dropout_rate'])
        self.activation = model_config['DeeperResNetPointNet'].get('activation', 'relu')

        if in_features != out_features:
            self.match_dimensions = nn.Linear(in_features, out_features)
        else:
            self.match_dimensions = None

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out) if self.activation == 'relu' else F.leaky_relu(out, negative_slope=0.1)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.match_dimensions:
            residual = self.match_dimensions(residual)

        out += residual
        return F.relu(out) if self.activation == 'relu' else F.leaky_relu(out, negative_slope=0.1)

class DeeperResNetPointNet(nn.Module):
    ### Deeper ResNet model for PointNet
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, num_classes, num_res_blocks=4):
        super(DeeperResNetPointNet, self).__init__()
        repo_path = '/content/DL_Bradford'
        with open(os.path.join(repo_path, 'configs/model_config.yaml'), 'r') as file:
            model_config = yaml.safe_load(file)

        print(f'Using dropout rate: {model_config["DeeperResNetPointNet"]["dropout_rate"]}')
        print(f'Using activation function: {model_config["DeeperResNetPointNet"]["activation"]}')

        self.num_classes = num_classes
        self.activation = model_config['DeeperResNetPointNet'].get('activation', 'relu')
        self.fc_initial = nn.Linear(input_size, hidden_layer1_size)
        self.bn_initial = nn.BatchNorm1d(hidden_layer1_size)
        self.dropout_initial = nn.Dropout(model_config['DeeperResNetPointNet']['dropout_rate'])
        
        self.res_blocks = nn.ModuleList([ResNetBlock(hidden_layer1_size, hidden_layer1_size) for _ in range(num_res_blocks - 1)])
        self.res_blocks.append(ResNetBlock(hidden_layer1_size, hidden_layer2_size))
        self.fc_final = nn.Linear(hidden_layer2_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.fc_initial(x)
        x = self.bn_initial(x)
        x = F.relu(x) if self.activation == 'relu' else F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout_initial(x)
        
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
