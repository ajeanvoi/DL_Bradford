import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.custom_dataset import PointCloudDataset
from models.my_model import PointNet
from models.modelV2 import ImprovedPointNet
from models.modelV3 import ResNetPointNet

def evaluate_model(model, dataloader, device):
    classes = {
        'arch': 0,
        'columns': 1,
        'moldings': 2,
        'floor': 3,
        'door & window': 4,
        'wall': 5,
        'stairs': 6,
        'vault': 7,
        'roof': 8,
        'other': 9,
    }

    # Reverse classes dictionary to get label names
    classes_reverse = {v: k for k, v in classes.items()}
    model_name = model.__class__.__name__
    model.eval()
    model.to(device)  # Move model to device

    # Initialize variables to track total and correct predictions
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_points = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_points.extend(inputs.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')

    # Check classes in y_true and y_pred
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    print(f'Unique labels in y_true: {unique_labels}')
    print(f'Unique labels in y_pred: {unique_preds}')

    # Create results directories if they don't exist
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    # Save results to CSV
    results_df = pd.DataFrame({
        'True': all_labels, 
        'Predicted': all_preds, 
        'x': [p[0] for p in all_points],
        'y': [p[1] for p in all_points],
        'z': [p[2] for p in all_points],
        'r': [p[3] for p in all_points],
        'g': [p[4] for p in all_points],
        'b': [p[5] for p in all_points],
    })
    results_df.to_csv('results/predictions/evaluation_results_'+ model_name + '.csv', index=False)

    # Plot and save the confusion matrix with percentages
    cm = confusion_matrix(all_labels, all_preds, labels=list(classes.values()))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='coolwarm', xticklabels=list(classes.keys()), yticklabels=list(classes.keys()))
    plt.title('Confusion Matrix (Normalized) : ' + model_name)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('results/visualizations/confusion_matrix_normalized_' + model_name + '.png')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    with open('configs/dataset_config.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    with open('configs/model_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    val_dataset = PointCloudDataset(dataset_config['dataset']['val_path'])
    val_loader = DataLoader(val_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=False)
    
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
    
    # Load the best model
    #model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.load_state_dict(torch.load('checkpoints/model_V3_augmented.pth')) # To modify the model name
    
    evaluate_model(model, val_loader, device)
