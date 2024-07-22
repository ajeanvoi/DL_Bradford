import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.custom_dataset import PointCloudDataset
from models.my_model import PointNet
from models.modelV2 import ImprovedPointNet
from models.modelV3 import ResNetPointNet

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    print('Training model:')
    print(model.__class__.__name__)

    model.to(device) 
    for epoch in range(num_epochs):
        start_time = time.time()
        
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time #Calculate epoch time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}, Time: {epoch_time:.2f} seconds')
    
    print('Finished Training')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    with open('configs/dataset_config.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    with open('configs/model_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    train_dataset = PointCloudDataset(dataset_config['dataset']['train_path'])
    train_loader = DataLoader(train_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=True)
    
    model_name = 'model_V3'
    print('Will be saved on checkpoints/' + model_name + '.pth')

    # model = PointNet(
    #     model_config['model_V1']['input_size'],
    #     model_config['model_V1']['hidden_layer1_size'],
    #     model_config['model_V1']['hidden_layer2_size'],
    #     model_config['model_V1']['num_classes']
    # )

    # model = ImprovedPointNet(
    #     model_config['model_V2']['input_size'],
    #     model_config['model_V2']['hidden_layer1_size'],
    #     model_config['model_V2']['hidden_layer2_size'],
    #     model_config['model_V2']['num_classes']
    # )

    model = ResNetPointNet(
        model_config['model_V3']['input_size'],
        model_config['model_V3']['hidden_layer1_size'],
        model_config['model_V3']['hidden_layer2_size'],
        model_config['model_V3']['num_classes']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config[model_name]['learning_rate'])
    
    train_model(model, train_loader, criterion, optimizer, model_config[model_name]['num_epochs'], device)
    
    torch.save(model.state_dict(), 'checkpoints/' + model_name + '.pth')
    print('Model saved to checkpoints/' + model_name + '.pth')
