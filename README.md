# My Pointcloud Project

Ce projet utilise Torch Points3D pour entraîner un modèle de segmentation de nuages de points afin de labelliser des bâtiments. Ce README décrit l'architecture du projet et fournit des instructions pour l'installation et l'utilisation.

## Structure du projet

```plaintext
my_pointcloud_project/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── custom_dataset.py
│
├── configs/
│   ├── dataset_config.yaml
│   └── model_config.yaml
│
├── models/
│   ├── __init__.py
│   └── my_model.py
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── checkpoints/
│   └── best_model.pth
│
├── logs/
│   └── training_logs/
│       └── log.txt
│
├── results/
│   ├── predictions/
│   └── visualizations/
│
├── requirements.txt
└── README.md

**Description des dossiers et fichiers**

- `data/`: Contient les données brutes et prétraitées ainsi que le script pour charger le dataset personnalisé.
    - `raw/`: Dossiers pour les données brutes d'entraînement, de validation et de test.
    - `processed/`: Dossiers pour les données prétraitées.
    - `custom_dataset.py`: Script pour créer une classe de dataset personnalisé compatible avec Torch Points3D.

- `configs/`: Contient les fichiers de configuration YAML pour le dataset et le modèle.
    - `dataset_config.yaml`: Configuration du dataset.
    - `model_config.yaml`: Configuration du modèle.

- `models/`: Contient les définitions des modèles.
    - `__init__.py`: Permet d'importer les modules du dossier.
    - `my_model.py`: Définition du modèle de segmentation personnalisé.

- `scripts/`: Contient les scripts pour les différentes étapes du projet.
    - `preprocess_data.py`: Script pour prétraiter les données brutes.
    - `train.py`: Script pour entraîner le modèle.
    - `evaluate.py`: Script pour évaluer les performances du modèle.
    - `predict.py`: Script pour faire des prédictions sur de nouvelles données.

- `notebooks/`: Contient des notebooks Jupyter pour l'exploration et la visualisation des données.
    - `data_exploration.ipynb`: Notebook pour l'exploration initiale des données.

- `checkpoints/`: Contient les checkpoints des modèles entraînés.
    - `best_model.pth`: Le meilleur modèle sauvegardé.

- `logs/`: Contient les logs des entraînements.
    - `training_logs/`: Sous-dossier pour stocker les logs d'entraînement.

- `results/`: Contient les résultats des prédictions et les visualisations.
    - `predictions/`: Les fichiers de prédictions générés par le modèle.
    - `visualizations/`: Les visualisations des résultats.

- `requirements.txt`: Liste des dépendances Python nécessaires pour le projet.

- `README.md`: Fichier décrivant le projet, comment l'installer, l'utiliser et toute autre information pertinente.


Installez les dépendances :

```plaintext
pip install -r requirements.txt
```

Pour utiliser CUDA de NVIDIA

```plaintext
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**Utilisation**

**Prétraitement des données**

Exécutez le script de prétraitement pour transformer les données brutes en données prêtes pour l'entraînement :

```plaintext
python scripts/preprocess_data.py
```

**Entraînement du modèle**

Lancez l'entraînement du modèle en utilisant le script d'entraînement :

```plaintext
python scripts/train.py
```

**Évaluation du modèle**

Évaluez les performances du modèle sur le dataset de validation :

```plaintext
python scripts/evaluate.py
```

**Prédictions**

Utilisez le modèle entraîné pour faire des prédictions sur de nouvelles données :

```plaintext
python scripts/predict.py
```

**Exploration des données**

Utilisez le notebook Jupyter pour explorer et visualiser vos données :

```plaintext
jupyter notebook notebooks/data_exploration.ipynb
```

**Contributeurs**

- Achille Jeanvoine - Développeur principal - [Mon GitHub](https://github.com/ajeanvoi)
- Crédits : 