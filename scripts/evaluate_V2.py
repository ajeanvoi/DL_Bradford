import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score
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
from models.modelV3_augmented import ResNetPointNet  # Utilisation de la version augmentée

# Import de la librairie pour la sigmoid focal loss
from scripts.sigmoidFocalLoss import SigmoidFocalLoss

def evaluate_model(model, dataloader, criterion, device, model_name):
    # classes = {
    #     'arch': 0,
    #     'columns': 1,
    #     'moldings': 2,
    #     'floor': 3,
    #     'door & window': 4,
    #     'wall': 5,
    #     'stairs': 6,
    #     'vault': 7,
    #     'roof': 8,
    #     'other': 9,
    # }

    classes = {'arch': 0,
           'columns': 1,
           'moldings': 2,
           'floor': 3,
           'window': 4,
           'wall': 5,
           'stairs': 6,
           'door': 7,
           'roof': 8,
           'other': 9,
           }

    # Reverse classes dictionary to get label names
    classes_reverse = {v: k for k, v in classes.items()}
    model.to(device)  # Move model to device
    model.eval()

    # Initialize variables to track total and correct predictions
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_points = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

            outputs = model(inputs)
            loss = criterion(outputs, nn.functional.one_hot(labels, num_classes=model.num_classes).float())
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_points.extend(inputs.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    avg_loss = running_loss / len(dataloader)

    print(f'Validation Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Check classes in y_true and y_pred
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    print(f'Unique labels in y_true: {unique_labels}')
    print(f'Unique labels in y_pred: {unique_preds}')

    # Filter out classes not present in the dataset
    present_classes = np.union1d(unique_labels, unique_preds)
    filtered_classes = {k: v for k, v in classes.items() if v in present_classes}
    filtered_classes_reverse = {v: k for k, v in filtered_classes.items()}

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
    results_df.to_csv('results/predictions/evaluation_results_' + model_name + '.csv', index=False)

    # Plot and save the confusion matrix with percentages
    cm = confusion_matrix(all_labels, all_preds, labels=list(filtered_classes.values()))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Reds', xticklabels=list(filtered_classes.keys()), yticklabels=list(filtered_classes.keys()))
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

    augmentations = None
    if dataset_config['augmentation']['enabled']:
        augmentations = dataset_config['augmentation']['augmentations']

    val_dataset = PointCloudDataset(dataset_config['dataset']['val_path'], augmentations=augmentations)
    val_loader = DataLoader(val_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=False)
    
    model = ResNetPointNet(
        model_config['model_V3_augmented']['input_size'],
        model_config['model_V3_augmented']['hidden_layer1_size'],
        model_config['model_V3_augmented']['hidden_layer2_size'],
        model_config['model_V3_augmented']['num_classes']
    )
    
    model_name = 'model_FL_alpha_0.422_gamma_1.528_BA_0.3945_F1_0.5887'
    checkpoint = torch.load('checkpoints/' + model_name + '.pth')
    model.load_state_dict(checkpoint)
    
    criterion = SigmoidFocalLoss(alpha=0.422, gamma=1.528, reduction='mean')
    
    evaluate_model(model, val_loader, criterion, device, model_name)
