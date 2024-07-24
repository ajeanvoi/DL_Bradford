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
import os
import sys

# Ajouter le chemin du dépôt au sys.path
repo_path = '/content/DL_Bradford'
sys.path.append(repo_path)

# Import des scripts spécifiques
from data.custom_dataset import PointCloudDataset
from models.modelV3_augmented import ResNetPointNet  # Utilisation de la version augmentée
from scripts.sigmoidFocalLoss import SigmoidFocalLoss

def evaluate_model(model, dataloader, criterion, device, model_name):
    classes = {
        'arch': 0,
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

    classes_reverse = {v: k for k, v in classes.items()}
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_points = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

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

    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    print(f'Unique labels in y_true: {unique_labels}')
    print(f'Unique labels in y_pred: {unique_preds}')

    present_classes = np.union1d(unique_labels, unique_preds)
    filtered_classes = {k: v for k, v in classes.items() if v in present_classes}
    filtered_classes_reverse = {v: k for k, v in filtered_classes.items()}

    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

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
    results_df.to_csv(f'results/predictions/evaluation_results_{model_name}.csv', index=False)

    cm = confusion_matrix(all_labels, all_preds, labels=list(filtered_classes.values()))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Reds', xticklabels=list(filtered_classes.keys()), yticklabels=list(filtered_classes.keys()))
    plt.title(f'Confusion Matrix (Normalized) : {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'results/visualizations/test_confusion_matrix_normalized_{model_name}.png')
    plt.show()

# Définir le dispositif et charger les configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

with open(os.path.join(repo_path, 'configs/dataset_config.yaml'), 'r') as file:
    dataset_config = yaml.safe_load(file)

with open(os.path.join(repo_path, 'configs/model_config.yaml'), 'r') as file:
    model_config = yaml.safe_load(file)

augmentations = None
if dataset_config['augmentation']['enabled']:
    augmentations = dataset_config['augmentation']['augmentations']

test_dataset = PointCloudDataset(dataset_config['dataset']['test_path'], augmentations=augmentations)
test_loader = DataLoader(test_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=False)

# Charger le modèle
model = ResNetPointNet(
    model_config['model_V3_augmented']['input_size'],
    model_config['model_V3_augmented']['hidden_layer1_size'],
    model_config['model_V3_augmented']['hidden_layer2_size'],
    model_config['model_V3_augmented']['num_classes']
)

model_name = 'model_FL_alpha_0.422_gamma_1.528_BA_0.3945_F1_0.5887'
checkpoint_path = os.path.join(repo_path, 'checkpoints', f'{model_name}.pth')
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

criterion = SigmoidFocalLoss(alpha=0.422, gamma=1.528, reduction='mean')

# Évaluer le modèle
evaluate_model(model, test_loader, criterion, device, model_name)
