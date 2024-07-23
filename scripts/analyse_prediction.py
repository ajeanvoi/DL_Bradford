import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_distribution(predictions_path, output_plot_path):
    # Charger les prédictions
    predictions_df = pd.read_csv(predictions_path)

    # Vérifier que la colonne 'Predicted' est présente
    if 'Predicted' not in predictions_df.columns:
        raise ValueError("Le fichier de prédictions doit contenir une colonne 'Predicted'.")

    # Calculer la distribution des classes
    class_counts = predictions_df['Predicted'].value_counts(normalize=True) * 100

    # Afficher la distribution des classes
    print(f"Class distribution (in %):\n{class_counts}")

    # Visualiser la distribution des classes
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution des classes prédites')
    plt.xlabel('Classes')
    plt.ylabel('Fréquence (%)')
    plt.xticks(rotation=45)
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

model_name = 'model_FL_alpha_0.250_gamma_2.000'

# Chemin vers le fichier de prédictions
predictions_path = '/content/results/predictions/predictions_unlabelled_' + model_name +'_.csv'

# Chemin pour sauvegarder le graphique
output_plot_path = '/content/results/predictions/class_distribution_' + model_name +'_.png'

# Analyser la distribution des classes
analyze_class_distribution(predictions_path, output_plot_path)
