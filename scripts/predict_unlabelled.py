import torch
import yaml
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader

# Ajouter le chemin du dépôt au sys.path
repo_path = '/content/DL_Bradford'
sys.path.append(repo_path)

# Import des scripts spécifiques
from data.custom_dataset import PointCloudDataset
# from models.modelV3_augmented import ResNetPointNet  # Utilisation de la version augmentée
from models.DeeperResNetPointNet import DeeperResNetPointNet  # Utilisation de la version augmentée

def load_raw_data(raw_data_path):
    # Lire les données brutes (non normalisées) depuis le fichier CSV
    raw_df = pd.read_csv(raw_data_path, sep=' ')
    # Extraire les coordonnées x, y, z, r, g, b
    points = raw_df[['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf']].values
    return points

def predict_and_save(model, dataloader, device, model_name, raw_data_path, output_path):
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

    # Utiliser model.predict
    model.eval()

    all_preds = []
    all_points = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            preds = model.predict(inputs)  # Utiliser la méthode predict

            all_preds.extend(preds.cpu().numpy())
            all_points.extend(inputs.cpu().numpy())

    # Charger les données brutes pour obtenir les coordonnées originales
    raw_points = load_raw_data(raw_data_path)

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame({
        '//X': raw_points[:, 0],
        'Y': raw_points[:, 1],
        'Z': raw_points[:, 2],
        'Rf': raw_points[:, 3],
        'Gf': raw_points[:, 4],
        'Bf': raw_points[:, 5],
        # # 'Predicted': [classes_reverse[p] for p in all_preds],
        'Predicted': all_preds,  # Utiliser les classes numériques
    })

    # Enregistrer les résultats
    results_df.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')

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

# Charger les données non étiquetées
test_dataset = PointCloudDataset(dataset_config['dataset']['predict_path'], augmentations=augmentations, has_labels=False)
test_loader = DataLoader(test_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=False)

# Charger le modèle
# model = ResNetPointNet(
#     model_config['model_V3_augmented']['input_size'],
#     model_config['model_V3_augmented']['hidden_layer1_size'],
#     model_config['model_V3_augmented']['hidden_layer2_size'],
#     model_config['model_V3_augmented']['num_classes']
# )

model = DeeperResNetPointNet(
    model_config['DeeperResNetPointNet']['input_size'],
    model_config['DeeperResNetPointNet']['hidden_layer1_size'],
    model_config['DeeperResNetPointNet']['hidden_layer2_size'],
    model_config['DeeperResNetPointNet']['num_classes'],
    model_config['DeeperResNetPointNet']['num_res_blocks']
)

# Charger les poids du modèle
model_name = 'V17_DeeperResNetPointNet_FL_alpha_0.250_gamma_2.000'
checkpoint_path = os.path.join(repo_path, 'checkpoints', f'{model_name}.pth')
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# Chemin de sortie pour les prédictions
output_path = 'results/predictions/predictions_unlabelled_' + model_name + '_.csv'

# Créer le dossier de sortie si nécessaire
os.makedirs(os.path.dirname('results/predictions/'), exist_ok=True)

# Chemin vers les données brutes pour les coordonnées
raw_data_path = dataset_config['dataset']['raw_path']

# Prédire et enregistrer les résultats
predict_and_save(model, test_loader, device, model_name, raw_data_path, output_path)
