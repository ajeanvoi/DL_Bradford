import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_data(csv_path):
    # True,Predicted,x,y,z,r,g,b
    return pd.read_csv(csv_path, sep=',', header=None, names=['True', 'Predicted', 'x', 'y', 'z', 'r', 'g', 'b'],encoding='utf-16')

def visualize_points(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normaliser les couleurs
    colors = df[['r', 'g', 'b']] / 255.0

    # Scatter plot des points avec les couleurs correspondantes
    scatter = ax.scatter(df['x'], df['y'], df['z'], c=colors, marker='o')

    # Ajout de la légende
    unique_classes = df['True'].unique()
    for cls in unique_classes:
        idx = df['True'] == cls
        ax.scatter(df['x'][idx], df['y'][idx], df['z'][idx], label=f'Classe {cls}', marker='o')

    ax.set_title('Points 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualiser les points 3D à partir d\'un fichier CSV.')
    parser.add_argument('csv_path', type=str, help='Chemin vers le fichier CSV contenant les points 3D.')
    args = parser.parse_args()

    # Charger les données
    df = load_data(args.csv_path)

    # Visualiser les points
    visualize_points(df)
