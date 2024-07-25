import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_points_kernels as tp

from torch_points3d.core.base_conv.dense import *
from torch_points3d.core.spatial_ops import DenseRadiusNeighbourFinder, DenseFPSSampler
from torch_points3d.utils.model_building_utils.activation_resolver import get_activation

__all__ = ["PointNetMSGDown", "PointNetPP"]

class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        bn=True,
        activation=torch.nn.LeakyReLU(negative_slope=0.01),
        use_xyz=True,
        normalize_xyz=False,
        **kwargs
    ):
        assert len(radii) == len(nsample) == len(down_conv_nn)
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )
        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            self.mlps.append(MLP2D(down_conv_nn[i], bn=bn, activation=activation, bias=False))
        self.radii = radii
        self.normalize_xyz = normalize_xyz

    def _prepare_features(self, x, pos, new_pos, idx, scale_idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(new_pos_trans, idx)  # (B, 3, npoint, nsample)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            grouped_pos /= self.radii[scale_idx]

        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_pos, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_pos

        return new_features

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, N, C]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        Returns:
            new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
        """
        assert scale_idx < len(self.mlps)
        new_features = self._prepare_features(x, pos, new_pos, radius_idx, scale_idx)
        new_features = self.mlps[scale_idx](new_features)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features


class PointNetPP(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPP, self).__init__()
        self.sa1 = PointNetMSGDown(npoint=512, radii=[0.1, 0.2], nsample=[16, 32], 
                                   down_conv_nn=[[3, 32, 32, 64], [3, 64, 64, 128]])
        self.sa2 = PointNetMSGDown(npoint=128, radii=[0.2, 0.4], nsample=[32, 64], 
                                   down_conv_nn=[[128, 128, 128, 256], [128, 256, 256, 512]])
        self.fc1 = nn.Linear(512 + 256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_points, _ = x.size()
        pos = x[..., :3]
        features = None
        
        new_pos, new_features = self.sa1(features, pos, pos)
        new_pos, new_features = self.sa2(new_features, new_pos, new_pos)

        x = new_features.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)


if __name__ == "__main__":
    # Nombre de classes pour la classification
    num_classes = 10

    # Initialiser le modèle
    model = PointNetPP(num_classes=num_classes)

    # Exemple de données d'entrée
    pos = torch.rand((16, 1024, 3))  # Batch de 16 nuages de points, chacun avec 1024 points et 3 coordonnées (x, y, z)

    # Passer les données à travers le modèle
    out = model(pos)
    print(out.shape)  # Devrait afficher torch.Size([16, num_classes])
