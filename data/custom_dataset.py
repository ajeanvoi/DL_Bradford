import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class PointCloudDataset(Dataset):
    def __init__(self, csv_file, augmentations=None, has_labels=True):
        self.data = pd.read_csv(csv_file, sep=',')
        print("Columns loaded from DataFrame : ", self.data.columns)
        if has_labels:
            self.data = self.data.dropna()
            # Vérifiez l'ordre des colonnes et affichez les premières lignes pour validation
            print(self.data.head())

        self.augmentations = augmentations
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # Vérifiez que les colonnes sont bien extraites
        # print("Sample Extracted: ", sample[['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Intensity',
        #                                     'Planarity_(0.2)', '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)',
        #                                     'Omnivariance_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)']])

        features = sample[['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf',
                           'Planarity_(0.2)', '2nd_eigenvalue_(0.2)', '3rd_eigenvalue_(0.2)',
                           'Omnivariance_(0.2)', 'Surface_variation_(0.2)', 'Sphericity_(0.2)', 'Verticality_(0.2)']].values
        #features = sample[['//X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'Intensity']].values
        
        features = features.astype(float)

         # Appliquer les augmentations si présentes
        features = self.augment_point_cloud(features)

        if self.has_labels:
            #print("Classification before processing: ", sample['Classification'])
            label = sample['Classification']
            try:
                label = int(label)
            except ValueError:
                raise ValueError(f"Label is not an integer at index {idx}: {label}")
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            #print("Features without labels: ", features)
            return torch.tensor(features, dtype=torch.float32)


    def augment_point_cloud(self, point_cloud):
        if self.augmentations is not None:
            for aug in self.augmentations:
                if aug['type'] == 'random_rotation':
                    point_cloud = self.random_rotation(point_cloud, aug['axis'], aug['angle_range'])
                elif aug['type'] == 'random_jitter':
                    point_cloud = self.random_jitter(point_cloud, aug['std'])
                elif aug['type'] == 'random_scale':
                    point_cloud = self.random_scale(point_cloud, aug['scale_range'])
                # elif aug['type'] == 'random_dropout':
                #     point_cloud = self.random_dropout(point_cloud, aug['drop_rate'])
        return point_cloud

    def random_rotation(self, point_cloud, axis, angle_range):
        theta = np.radians(random.uniform(angle_range[0], angle_range[1]))
        if axis == 'z':
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
        else:
            raise ValueError('Unsupported rotation axis')
        # point_cloud[:, :3] = np.dot(point_cloud[:, :3], rotation_matrix)
        point_cloud[:3] = np.dot(point_cloud[:3], rotation_matrix)
        return point_cloud

    def random_jitter(self, point_cloud, std):
        #jitter = np.random.normal(0, std, point_cloud[:, :3].shape)
        jitter = np.random.normal(0, std, point_cloud[:3].shape)
        # point_cloud[:, :3] += jitter
        point_cloud[:3] += jitter
        return point_cloud

    def random_scale(self, point_cloud, scale_range):
        scale = random.uniform(scale_range[0], scale_range[1])
        #point_cloud[:, :3] *= scale
        point_cloud[:3] *= scale
        return point_cloud

    # not working
    # def random_dropout(self, point_cloud, drop_rate):
    #     num_points = point_cloud.shape[0]
    #     num_drop = int(num_points * drop_rate)
    #     drop_indices = np.random.choice(num_points, num_drop, replace=False)
    #     point_cloud = np.delete(point_cloud, drop_indices, axis=0)
    #     return point_cloud
