import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_points_kernels as tp

from torch_points3d.core.base_conv.dense import BaseDenseConvolutionDown
from torch_points3d.core.common_modules.dense_modules import MLP2D
from torch_points3d.core.spatial_ops import DenseFPSSampler, DenseRadiusNeighbourFinder

log = logging.getLogger(__name__)

class RSConvMapper(nn.Module):
    def __init__(self, down_conv_nn, use_xyz, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        """
        RSConvMapper is used to map and transform features and messages in the RSConv architecture.
        
        Args:
            down_conv_nn: List of tuples defining the input, intermediate, and output feature dimensions.
            use_xyz: Boolean indicating whether to use xyz coordinates in the features.
            bn: Boolean indicating whether to use batch normalization.
            activation: Activation function to be used after normalization.
        """
        super(RSConvMapper, self).__init__()

        self._down_conv_nn = down_conv_nn
        self._use_xyz = use_xyz

        self.nn = nn.ModuleDict()

        if len(self._down_conv_nn) == 2:  # Check if it's the first layer
            self._first_layer = True
            f_in, f_intermediate, f_out = self._down_conv_nn[0]
            # First layer: Apply MLP2D for feature transformation
            self.nn["features_nn"] = MLP2D(self._down_conv_nn[1], bn=bn, bias=False)
        else:
            self._first_layer = False
            f_in, f_intermediate, f_out = self._down_conv_nn

        # Define the MLP to process messages
        self.nn["mlp_msg"] = MLP2D([f_in, f_intermediate, f_out], bn=bn, bias=False)
        # Batch normalization and activation function
        self.nn["norm"] = nn.Sequential(nn.BatchNorm2d(f_out), activation)

        self._f_out = f_out

    @property
    def f_out(self):
        return self._f_out

    def forward(self, features, msg):
        """
        Forward pass of RSConvMapper.
        
        Args:
            features: Tensor of shape [B, C, num_points, nsamples].
            msg: Tensor of shape [B, 10, num_points, nsamples] containing distance, coordinates, and deltas.

        Returns:
            Processed features after applying mapping and normalization.
        """
        # Transform message using MLP
        msg = self.nn["mlp_msg"](msg)
        # Apply the feature transformation if it's the first layer
        if self._first_layer:
            features = self.nn["features_nn"](features)
        # Element-wise multiplication followed by normalization
        return self.nn["norm"](torch.mul(features, msg))

class SharedRSConv(nn.Module):
    def __init__(self, mapper: RSConvMapper, radius):
        """
        SharedRSConv applies the RSConvMapper to aggregate and transform features.

        Args:
            mapper: An instance of RSConvMapper used for feature and message processing.
            radius: The radius for neighborhood search.
        """
        super(SharedRSConv, self).__init__()
        self._mapper = mapper
        self._radius = radius

    def forward(self, aggr_features, centroids):
        """
        Forward pass of SharedRSConv.

        Args:
            aggr_features: Aggregated features including coordinates and distances.
            centroids: Tensor of shape [B, 3, num_points, 1] representing the centroids of the neighborhoods.

        Returns:
            Processed features after applying RSConvMapper.
        """
        abs_coord = aggr_features[:, :3]  # Absolute coordinates
        delta_x = aggr_features[:, 3:6]  # Normalized coordinates
        features = aggr_features[:, 3:]  # Features after coordinates

        nsample = abs_coord.shape[-1]
        coord_xi = centroids.repeat(1, 1, 1, nsample)  # Repeat centroids for each sample

        distance = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)  # Calculate Euclidean distance
        h_xi_xj = torch.cat((distance, coord_xi, abs_coord, delta_x), dim=1)  # Concatenate features

        return self._mapper(features, h_xi_xj)  # Process features with RSConvMapper

    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self._radius})"

class RSConvSharedMSGDown(BaseDenseConvolutionDown):
    def __init__(self, npoint=None, radii=None, nsample=None, down_conv_nn=None, channel_raising_nn=None, bn=True, use_xyz=True, activation=nn.ReLU()):
        """
        RSConvSharedMSGDown performs downsampling and feature aggregation using shared RSConv layers.

        Args:
            npoint: Number of points to sample after downsampling.
            radii: List of radii for neighborhood search at different scales.
            nsample: List of number of samples in each neighborhood.
            down_conv_nn: List defining the MLP structure for down-convolution layers.
            channel_raising_nn: List defining the MLP structure for channel raising.
            bn: Boolean indicating whether to use batch normalization.
            use_xyz: Boolean indicating whether to include xyz coordinates in the features.
            activation: Activation function to be used in MLPs.
        """
        assert len(radii) == len(nsample)
        if len(radii) != len(down_conv_nn):
            log.warning("The down_conv_nn has a different size as radii. Make sure of have SharedRSConv")
        super(RSConvSharedMSGDown, self).__init__(DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample))

        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()

        self._mapper = RSConvMapper(down_conv_nn, activation=activation, use_xyz=self.use_xyz)

        self.mlp_out = nn.Sequential(
            nn.Conv1d(channel_raising_nn[0], channel_raising_nn[-1], kernel_size=1, stride=1, bias=True),
            nn.BatchNorm1d(channel_raising_nn[-1]),
            activation,
        )

        for i in range(len(radii)):
            self.mlps.append(SharedRSConv(self._mapper, radii[i]))

    def _prepare_features(self, x, pos, new_pos, idx):
        """
        Prepare features for convolution by grouping and concatenating positional and feature information.

        Args:
            x: Tensor of features [B, N, C].
            pos: Tensor of positions [B, N, 3].
            new_pos: Tensor of new positions [B, npoint, 3].
            idx: Indices for grouping [B, npoint, nsample].

        Returns:
            New features and centroids after grouping and concatenation.
        """
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos_absolute = tp.grouping_operation(new_pos_trans, idx)
        centroids = new_pos.transpose(1, 2).unsqueeze(-1)
        grouped_pos_normalized = grouped_pos_absolute - centroids

        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_pos_absolute, grouped_pos_normalized, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([grouped_pos_absolute, grouped_pos_normalized], dim=1)

        return new_features, centroids

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """
        Perform a convolution operation by aggregating features and applying RSConv layers.

        Args:
            x: Previous features [B, N, C].
            pos: Previous positions [B, N, 3].
            new_pos: Sampled positions [B, npoints, 3].
            radius_idx: Indices for grouping [B, npoints, nsample].
            scale_idx: Index for selecting the scale of the RSConv layer.

        Returns:
            New features after convolution and pooling.
        """
        assert scale_idx < len(self.mlps)
        aggr_features, centroids = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self.mlps[scale_idx](aggr_features, centroids)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        new_features = self.mlp_out(new_features.squeeze(-1))
        return new_features

    def __repr__(self):
        return f"{self.__class__.__name__}(mlps={self.mlps.__repr__()}, mapper={self._mapper.__repr__()})"
