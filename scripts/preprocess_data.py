import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# Seperation of door and window, remove vault
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
        df = pd.read_csv(file_path)
        data_frames.append(df)
        print(f'Loaded {file_name}, columns: {df.columns}')
    data = pd.concat(data_frames, ignore_index=True)
    return data

def plot_class_distribution(data, title, output_path):
    plt.figure(figsize=(10, 6))
    if 'Classification' in data.columns:
        data['ClassName'] = data['Classification'].map(class_names)
        class_counts = data['ClassName'].value_counts(normalize=True) * 100
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Frequency (%)')
        plt.xticks(rotation=45)
        for i, v in enumerate(class_counts.values):
            plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        print(f"Error: 'Classification' column not found in data for {title}")

def preprocess_file(file_path, columns_to_keep):
    data = pd.read_csv(file_path, skiprows=1, header=None, sep=' ')
    column_names = ['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Original_cloud_index'
                    'Planarity_(0.1)', 'Classification', 'Surface_variation_(0.1)', 'Verticality_(0.1)',
                    'PCA1_(0.1)', 'PCA2_(0.1)', '1st_eigenvalue_(0.2)', '2nd_eigenvalue_(0.2)', 
                    '3rd_eigenvalue_(0.2)', 'Number_of_neighbors_(r=1)', 'Omnivariance_(0.2)', 
                    'Planarity_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)', 
                    'Nx', 'Ny', 'Nz']
    data.columns = column_names
    data = data[columns_to_keep]
    data = data.dropna()
    data[['//X', 'Y', 'Z']] = (data[['//X', 'Y', 'Z']] - data[['//X', 'Y', 'Z']].mean()) / data[['//X', 'Y', 'Z']].std()
    data[['Rf', 'Gf', 'Bf']] = (data[['Rf', 'Gf', 'Bf']] - data[['Rf', 'Gf', 'Bf']].mean()) / data[['Rf', 'Gf', 'Bf']].std()
    data['//X'] = data['//X'].apply(lambda x: round(x, 4))
    data['Y'] = data['Y'].apply(lambda x: round(x, 4))
    data['Z'] = data['Z'].apply(lambda x: round(x, 4))
    data['Rf'] = data['Rf'].apply(lambda x: round(x, 3))
    data['Gf'] = data['Gf'].apply(lambda x: round(x, 3))
    data['Bf'] = data['Bf'].apply(lambda x: round(x, 3))
    return data

def balance_classes(data):
    if 'Classification' not in data.columns:
        print("Error: 'Classification' column not found in data")
        return data

    class_counts = data['Classification'].value_counts()
    majority_class = class_counts.idxmax()
    minority_classes = class_counts[class_counts < class_counts[majority_class]].index
    balanced_data = data[data['Classification'] == majority_class]
    for minority_class in minority_classes:
        minority_data = data[data['Classification'] == minority_class]
        minority_data_upsampled = resample(minority_data, 
                                           replace=True,     
                                           n_samples=len(balanced_data) // len(minority_classes),   
                                           random_state=123)
        balanced_data = pd.concat([balanced_data, minority_data_upsampled])
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    return balanced_data

def preprocess_data(input_dir, output_dir, balance=False, isTesting = False):

    if (isTesting):
        columns_to_keep = ['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 
                       'Planarity_(0.2)',
                       '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)',
                       'Omnivariance_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)']
    else:
        columns_to_keep = ['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 
                       'Planarity_(0.2)', 'Classification',
                       '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)',
                       'Omnivariance_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)'] 
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        data = preprocess_file(file_path, columns_to_keep)
        if balance:
            data = balance_classes(data)
        output_path = os.path.join(output_dir, file_name)
        data.to_csv(output_path, index=False)

if __name__ == "__main__":
    visualization_dir = 'results/visualizations'
    os.makedirs(visualization_dir, exist_ok=True)
    
    # print('Preprocessing and visualizing training data...')
    # raw_train_data = load_data('data/raw/train')
    # plot_class_distribution(raw_train_data, 'Raw Training Data Class Distribution', 
    #                         os.path.join(visualization_dir, 'raw_training_data_distribution.png'))

    # preprocess_data('data/raw/train', 'data/processed/train', balance=True)
    # processed_train_data = load_data('data/processed/train')
    # plot_class_distribution(processed_train_data, 'Processed Training Data Class Distribution (Balanced)', 
    #                         os.path.join(visualization_dir, 'processed_training_data_distribution.png'))
    
    # print('Preprocessing and visualizing validation data...')
    # raw_val_data = load_data('data/raw/val')
    # plot_class_distribution(raw_val_data, 'Raw Validation Data Class Distribution', 
    #                         os.path.join(visualization_dir, 'raw_validation_data_distribution.png'))
    
    # preprocess_data('data/raw/val', 'data/processed/val')
    # processed_val_data = load_data('data/processed/val')
    # plot_class_distribution(processed_val_data, 'Processed Validation Data Class Distribution', 
    #                         os.path.join(visualization_dir, 'processed_validation_data_distribution.png'))

    print('Preprocessing test data...')
    #raw_test_data = load_data('data/raw/test')
    # plot_class_distribution(raw_test_data, 'Raw Test Data Class Distribution', 
    #                         os.path.join(visualization_dir, 'raw_test_data_distribution.png'))
    
    preprocess_data('data/raw/test', 'data/processed/test', isTesting = True)
    # processed_test_data = load_data('data/processed/test')
    # plot_class_distribution(processed_test_data, 'Processed Test Data Class Distribution', 
    #                         os.path.join(visualization_dir, 'processed_test_data_distribution.png'))
