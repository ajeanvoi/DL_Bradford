import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    """Load data from the CSV file."""
    return pd.read_csv(csv_path, sep=" ", header=None, skiprows=1, on_bad_lines='skip')


def split_data(data):
    """Split data into training, validation, and test sets."""
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, shuffle=False)
    return train_data, val_data, test_data

def rename_columns(data):
    """Rename columns in the DataFrame."""
    column_names = [
        '//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Original_cloud_index',
        'Classification', 'Surface_variation_(0.1)', 'Verticality_(0.1)',
        'Planarity_(0.1)', 'PCA2_(0.1)', 'PCA1_(0.1)', '1st_eigenvalue_(0.2)',
        '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)', 'Omnivariance_(0.2)',
        'Planarity_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)',
        'Verticality_(0.2)', 'Nx', 'Ny', 'Nz'
    ]
    data.rename(columns=dict(enumerate(column_names)), inplace=True)
    return data

def save_data(data, output_path, prefix):
    """Save data to CSV files."""
    os.makedirs(output_path, exist_ok=True)
    data.to_csv(os.path.join(output_path, f"{prefix}.csv"), index=False)

if __name__ == "__main__":

    # Paths for the data
    csv_path = "DL_Bradford/data/raw/train/V12_UK_dropNaN_train.csv"
    output_path = "DL_Bradford/data/rawSplit"

    # Les datas seront dans 3 sous dossiers: train, validation et test
    # Création de ces sous dossiers :
    train_path = os.path.join(output_path, "train")
    os.makedirs(train_path, exist_ok=True)
    val_path = os.path.join(output_path, "val")
    os.makedirs(val_path, exist_ok=True)
    test_path = os.path.join(output_path, "test")
    os.makedirs(test_path, exist_ok=True)

    data = load_data(csv_path)
    print(f"Loaded data with {len(data)} rows.")

    train_data, val_data, test_data = split_data(data)

    print("Column names before renaming:")
    print(data.columns.tolist())

    # train_data = rename_columns(train_data)
    # val_data = rename_columns(val_data)
    # test_data = rename_columns(test_data)

    print("\nColumn names after renaming:")
    print(train_data.columns.tolist())

    save_data(train_data, train_path, "train_modif")
    save_data(val_data, val_path, "validation_modif")
    save_data(test_data, test_path, "test_modif")

    # Print the number of rows in each file
    print(f"\nNumber of rows in train set: {len(train_data)}")
    print(f"Number of rows in validation set: {len(val_data)}")
    print(f"Number of rows in test set: {len(test_data)}")
    print("\nColumns renamed and data saved successfully on data/rawSplit folder.")
