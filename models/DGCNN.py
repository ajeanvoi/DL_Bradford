#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Inspired by the DGCNN model from the paper "Dynamic Graph CNN for Learning on Point Clouds" by Yue Wang.
@Author: Yue Wang
@Contact: yuewangx@mit.edu
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Utility function to load YAML configuration
def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Function to calculate k-nearest neighbors
def knn(x, k):
    """Calculate the k-nearest neighbors for each point in the point cloud."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


# Function to get graph feature
def get_graph_feature(x, k=20, idx=None):
    """Compute the graph feature for each point by concatenating the point with its k nearest neighbors."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


# DGCNN Model definition
class DGCNN(nn.Module):
    def __init__(self, config_path='DL_Bradford/configs/model_config.yaml', output_channels=None):
        """Initialize the DGCNN model with the parameters from the YAML configuration."""
        super(DGCNN, self).__init__()
        
        # Load the configuration
        config = load_config(config_path)
        self.k = config['DGCNN']['k']
        self.emb_dims = config['DGCNN']['emb_dims']
        self.dropout_rate = config['DGCNN']['dropout_rate']
        input_size = config['DGCNN']['input_size']
        self.num_classes = config['DGCNN']['num_classes']

        # Set the output channels if not provided
        self.output_channels = output_channels if output_channels else config['DGCNN']['num_classes']

        # Define the layers with flexible input sizes
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * input_size, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Fully connected layers for classification
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout_rate)
        
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout_rate)
        
        self.linear3 = nn.Linear(256, self.output_channels)

    def forward(self, x):
        """Forward pass through the DGCNN model."""
        batch_size = 256

        # Get the graph feature and apply the first convolution
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # Repeat for the other convolutional layers
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # Concatenate all layers
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Apply the final convolutional layer
        x = self.conv5(x)
        
        # Adaptive pooling
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Fully connected layers for classification
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x