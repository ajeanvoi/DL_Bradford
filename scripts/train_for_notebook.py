import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
import time
import random
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
import numpy as np

import os
import sys
# Ajouter le chemin du dépôt cloné au sys.path
repo_path = '/content/DL_Bradford'
sys.path.append(repo_path)

# Import des scripts spécifiques
from scripts.sigmoidFocalLoss import SigmoidFocalLoss
from data.custom_dataset import PointCloudDataset
from models.my_model import PointNet
from models.modelV2 import ImprovedPointNet
# from models.modelV3_augmented import ResNetPointNet  # Remove augmented if you want to use the non-augmented version
from models.model_V4 import DeeperResNetPointNet


# Utilisation : python scripts/train_with_augmentation.py --alpha 0.285 --gamma 2.158

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    print('Training model:')
    print(model.__class__.__name__)

    model.to(device)
    prev_loss = 1000000

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = nn.functional.one_hot(labels, num_classes=model.num_classes).float()
            print(f'Input shape: {inputs.shape}')
            print(f'Labels shape: {labels.shape}')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}, Time: {epoch_time:.2f} seconds')

        if running_loss / len(dataloader) > prev_loss:
            print('Loss did not decrease, stopping training')
            break

        prev_loss = running_loss / len(dataloader)

    print('Finished Training')

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, nn.functional.one_hot(labels, num_classes=model.num_classes).float())
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    avg_loss = running_loss / len(dataloader)

    return avg_loss, accuracy, f1, balanced_acc, precision, recall

## Random search for hyperparameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model with optional hyperparameter search.')
    parser.add_argument('--alpha', type=float, help='Alpha value for Sigmoid Focal Loss.')
    parser.add_argument('--gamma', type=float, help='Gamma value for Sigmoid Focal Loss.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    with open(os.path.join(repo_path, 'configs/dataset_config.yaml'), 'r') as file:
        dataset_config = yaml.safe_load(file)

    with open(os.path.join(repo_path, 'configs/model_config.yaml'), 'r') as file:
        model_config = yaml.safe_load(file)

    augmentations = None
    if dataset_config['augmentation']['enabled']:
        augmentations = dataset_config['augmentation']['augmentations']

    train_dataset = PointCloudDataset(dataset_config['dataset']['train_path'], augmentations=augmentations)
    train_loader = DataLoader(train_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=True)
    
    val_dataset = PointCloudDataset(dataset_config['dataset']['val_path'], augmentations=augmentations)
    val_loader = DataLoader(val_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=False)

    # Parameters for the model
    model_name = 'model_V4' # Change this to the model you want to use
    save_path = os.path.join(repo_path, 'checkpoints/')
    os.makedirs(save_path, exist_ok=True)

    print('Models will be saved to:', save_path)

    # Define random search parameters
    num_combinations = 5
    alpha_range = (0.1, 0.5)
    gamma_range = (1, 5)
    num_epochs = model_config[model_name]['num_epochs']
    best_f1 = 0
    best_params = {}
    isSearching = not (args.alpha is not None and args.gamma is not None)

    if not isSearching:
        # Use provided alpha and gamma values
        alpha = args.alpha
        gamma = args.gamma
        num_combinations = 1
        print(f'Using provided alpha={alpha} and gamma={gamma} for training.')
    else:
        alpha = None
        gamma = None
        print('No alpha and gamma provided, performing random search.')

    for _ in range(num_combinations):
        if alpha is None and gamma is None:
            alpha = random.uniform(*alpha_range)
            gamma = random.uniform(*gamma_range)
        
        print(f'Training with alpha={alpha:.3f}, gamma={gamma:.3f}')
        # model = ResNetPointNet(
        #     model_config['model_V3_augmented']['input_size'],
        #     model_config['model_V3_augmented']['hidden_layer1_size'],
        #     model_config['model_V3_augmented']['hidden_layer2_size'],
        #     model_config['model_V3_augmented']['num_classes']
        # )

        model = DeeperResNetPointNet(
            model_config['model_V4']['input_size'],
            model_config['model_V4']['hidden_layer1_size'],
            model_config['model_V4']['hidden_layer2_size'],
            model_config['model_V4']['num_classes'],
            model_config['model_V4']['num_res_blocks']
        )

        # model = PointNetPP(num_classes=model_config['model_V5']['num_classes'])

        criterion = SigmoidFocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=model_config[model_name]['learning_rate'])
        
        train_model(model, train_loader, criterion, optimizer, num_epochs, device)
        
        val_loss, val_accuracy, val_f1, val_balanced_acc, val_precision, val_recall = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}')

        # Save each model with alpha, gamma, balanced_accuracy, and f1 in the filename
        model_save_path = os.path.join(
            # save_path, f'model_FL_alpha_{alpha:.3f}_gamma_{gamma:.3f}_BA_{val_balanced_acc:.4f}_F1_{val_f1:.4f}.pth')
            save_path, f'model_V4_FL_alpha_{alpha:.3f}_gamma_{gamma:.3f}.pth')
        torch.save(model.state_dict(), model_save_path)
        print('Model saved to ' + model_save_path)
        
        if val_f1 > best_f1 and isSearching:
            best_f1 = val_f1
            best_params = {'alpha': alpha, 'gamma': gamma, 'loss': val_loss, 'accuracy': val_accuracy, 'f1': val_f1, 'balanced_acc': val_balanced_acc, 'precision': val_precision, 'recall': val_recall}
            best_model_save_path = os.path.join(save_path, f'best_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print('New best model found with F1:', val_f1)

    print('Best params:', best_params)
    print('Model saved to ' + best_model_save_path)

    if isSearching:
        print('\nNo specific alpha and gamma were provided, random search was performed.')
        print('You can also provide specific values for alpha and gamma like this:')
        print('python scripts/train_with_augmentation.py --alpha 0.285 --gamma 2.158')
