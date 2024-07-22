import torch
from torch.utils.data import DataLoader
import yaml
import pandas as pd
from data.custom_dataset import PointCloudDataset
from models.my_model import PointNet

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.numpy())
    return predictions

if __name__ == "__main__":
    with open('configs/dataset_config.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    with open('configs/model_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    test_dataset = PointCloudDataset(dataset_config['dataset']['test_path'])
    test_loader = DataLoader(test_dataset, batch_size=dataset_config['dataset']['batch_size'], shuffle=False)
    
    model = PointNet(
        model_config['model']['input_size'],
        model_config['model']['hidden_layer1_size'],
        model_config['model']['hidden_layer2_size'],
        model_config['model']['num_classes']
    )
    
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    
    predictions = predict(model, test_loader)
    
    output_file = 'results/predictions/predictions.csv'
    pd.DataFrame(predictions, columns=['prediction']).to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')
