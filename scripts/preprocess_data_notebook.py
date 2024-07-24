import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# Séparation de la porte et de la fenêtre, suppression du vault
class_names = {'arch': 0,
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

def load_data(input_dir):
    data_frames = []
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isdir(file_path):
            print(f'Skipping directory: {file_path}')
            continue
        print(f'Loading file: {file_path}')
        df = pd.read_csv(file_path)
        print(f'Loaded {file_name}, columns: {df.columns}')
        data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    print(f'Combined data shape: {data.shape}')
    return data

# def plot_class_distribution(data, title, output_path):
#     plt.figure(figsize=(10, 6))
#     if 'Classification' in data.columns:
#         print(f'Plotting class distribution for {title}')
#         data['ClassName'] = data['Classification'].map(class_names)
#         class_counts = data['ClassName'].value_counts(normalize=True) * 100
#         print(f'Class distribution:\n{class_counts}')
#         sns.barplot(x=class_counts.index, y=class_counts.values)
#         plt.title(title)
#         plt.xlabel('Classes')
#         plt.ylabel('Frequency (%)')
#         plt.xticks(rotation=45)
#         for i, v in enumerate(class_counts.values):
#             plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom')
#         plt.tight_layout()
#         plt.savefig(output_path)
#         print(f'Saved plot to {output_path}')
#         plt.close()
#     else:
#         print(f"Error: 'Classification' column not found in data for {title}")

def preprocess_file(file_path, columns_to_keep):
    print(f'Preprocessing file: {file_path}')
    data = pd.read_csv(file_path, skiprows=1, header=None, sep=' ')
    print(f'Initial data shape: {data.shape}')
    print(f'Initial columns: {data.columns.tolist()}')
    
    column_names = ['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Original_cloud_index',
                    'Classification', 'Surface_variation_(0.1)', 'Verticality_(0.1)',
                    'Planarity_(0.1)', 'PCA2_(0.1)', 'PCA1_(0.1)', '1st_eigenvalue_(0.2)', 
                    '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)', 'Omnivariance_(0.2)', 
                    'Planarity_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 
                    'Verticality_(0.2)', 'Nx', 'Ny', 'Nz']
    
    # Vérifier si le nombre de colonnes correspond
    if len(data.columns) != len(column_names):
        print(f"Warning: Number of columns in data ({len(data.columns)}) does not match column names list ({len(column_names)}). Adjusting column names list.")
    
    # Ajuster les noms des colonnes en fonction du nombre réel de colonnes
    data.columns = column_names[:len(data.columns)]
    print(f'Columns after renaming: {data.columns.tolist()}')
    
    data = data[columns_to_keep]
    print(f'Data shape after selecting columns: {data.shape}')
    #data = data.dropna()

    # Modification sur les données raw
    data.fillna(method='ffill', inplace=True)
    data.reset_index(drop=True, inplace=True)
    # data.dropna(inplace=True)
    # data.reset_index(drop=True, inplace=True)

    print(f'Data shape after dropping NA: {data.shape}')
    
    # Normaliser les coordonnées et les couleurs
    data[['//X', 'Y', 'Z']] = (data[['//X', 'Y', 'Z']] - data[['//X', 'Y', 'Z']].mean()) / data[['//X', 'Y', 'Z']].std()
    data[['Rf', 'Gf', 'Bf']] = (data[['Rf', 'Gf', 'Bf']] - data[['Rf', 'Gf', 'Bf']].mean()) / data[['Rf', 'Gf', 'Bf']].std()
    
    # Arrondir les valeurs
    data[['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf']] = data[['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf']].round()
    print(f'Preprocessed data shape: {data.shape}')
    
    return data

def balance_classes(data):
    if 'Classification' not in data.columns:
        print("Error: 'Classification' column not found in data")
        return data

    class_counts = data['Classification'].value_counts()
    print(f'Class distribution before balancing:\n{class_counts}')
    majority_class = class_counts.idxmax()
    minority_classes = class_counts[class_counts < class_counts[majority_class]].index
    print(f'Majority class: {majority_class}')
    print(f'Minority classes: {minority_classes}')
    
    balanced_data = data[data['Classification'] == majority_class]
    for minority_class in minority_classes:
        minority_data = data[data['Classification'] == minority_class]
        print(f'Upsampling minority class: {minority_class}, original size: {len(minority_data)}')
        minority_data_upsampled = resample(minority_data, 
                                           replace=True,     
                                           n_samples=len(balanced_data) // len(minority_classes),   
                                           random_state=123)
        print(f'Upsampled size: {len(minority_data_upsampled)}')
        balanced_data = pd.concat([balanced_data, minority_data_upsampled])
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    print(f'Balanced data shape: {balanced_data.shape}')
    return balanced_data

def preprocess_data(input_dir, output_dir, balance=False, isTesting=False):
    if isTesting:
        columns_to_keep = ['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 
                           'Planarity_(0.2)',
                           '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)',
                           'Omnivariance_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)']
    else:
        columns_to_keep = ['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 
                           'Planarity_(0.2)', 'Classification',
                           '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)',
                           'Omnivariance_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)']
    
    print(f'Processing files in {input_dir}, saving to {output_dir}, balance={balance}, isTesting={isTesting}')
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isdir(file_path):
            print(f'Skipping directory: {file_path}')
            continue
        print(f'Processing file: {file_path}')
        data = preprocess_file(file_path, columns_to_keep)
        if balance:
            print(f'Balancing classes for file: {file_name}')
            data = balance_classes(data)
        output_path = os.path.join(output_dir, file_name)
        data.to_csv(output_path, index=False)
        print(f'Saved preprocessed data to {output_path}')

if __name__ == "__main__":
    # Définir les chemins en fonction de l'environnement Colab
    base_dir = '/content/DL_Bradford'
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    visualization_dir = os.path.join(base_dir, 'results/visualizations')

    os.makedirs(visualization_dir, exist_ok=True)

    # Exemple de prétraitement et visualisation des données de test
    print('Preprocessing test data...')
    preprocess_data(os.path.join(raw_data_dir, 'test'), os.path.join(processed_data_dir, 'test'), isTesting=True)
