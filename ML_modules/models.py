import torch
import numpy as np
import open3d as o3d
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim: int, num_patch: int, token_dim: int, channel_dim: int, dropout: float = 0.):
        super(MixerBlock, self).__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)

        return x

class SimpleMixer(nn.Module):
    def __init__(self, dim: int, num_patch: int, embed_dim: int, depth: int, token_dim: int, 
                 channel_dim: int, dropout: float, num_classes: int):
        super(SimpleMixer, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(dim, embed_dim))
        mixers = [MixerBlock(embed_dim, num_patch, token_dim, channel_dim, dropout) for _ in range(depth)]
        self.backbone = nn.Sequential(*mixers)
        self.mlp_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'), 
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, num_classes)
            )

    def forward(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.mlp_head(x)

        return x
    
class LLGBlock(nn.Module):
    def __init__(self, dim: int, embed_dim: int):
        super(LLGBlock, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(dim, embed_dim), 
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.embedding(x)

        return x
    
class MSG_fpfh():

    def gen_features(self, pts_R: np.ndarray, pts_L: np.ndarray, normals_R: np.ndarray,
                     normals_L: np.ndarray) -> np.ndarray:
        """
        Generate features of the surface patches using Multi-scale Grouping 
        and Fast Point Feature Histograms.

        Parameters
        ----------
        pts_R : Nx3 : obj : `numpy.ndarray`
            pointcloud data of a right contact surface
        pts_L : Nx3 : obj : `numpy.ndarray`
            pointcloud data of a left contact surface
        normals_R : Nx3 : obj : `numpy.ndarray`
            surface normals of a right contact surface
        normals_L : Nx3 : obj : `numpy.ndarray`
            surface normals of a left contact surface
        
        Returns
        -------
        res : 105 : obj : `numpy.ndarray` 
            generated MSG_fpfh features
        """
        temp_1 = torch.from_numpy(pts_R).unsqueeze_(0)
        temp_2 = torch.from_numpy(pts_L).unsqueeze_(0)
        anchor_1 = self.farthest_point_sample(xyz=temp_1, npoint=20)[0].tolist()
        anchor_2 = self.farthest_point_sample(xyz=temp_2, npoint=20)[0].tolist()

        surf_1 = o3d.geometry.PointCloud()
        surf_1.points = o3d.utility.Vector3dVector(pts_R)
        surf_1.normals = o3d.utility.Vector3dVector(normals_R)

        surf_2 = o3d.geometry.PointCloud()
        surf_2.points = o3d.utility.Vector3dVector(pts_L)
        surf_2.normals = o3d.utility.Vector3dVector(normals_L)

        fpfh_11 = o3d.pipelines.registration.compute_fpfh_feature(surf_1,
                  o3d.geometry.KDTreeSearchParamRadius(radius=5)).data.T
        fpfh_12 = o3d.pipelines.registration.compute_fpfh_feature(surf_1,
                  o3d.geometry.KDTreeSearchParamRadius(radius=10)).data.T
        fpfh_13 = o3d.pipelines.registration.compute_fpfh_feature(surf_1,
                  o3d.geometry.KDTreeSearchParamRadius(radius=20)).data.T

        fpfh_21 = o3d.pipelines.registration.compute_fpfh_feature(surf_2,
                  o3d.geometry.KDTreeSearchParamRadius(radius=5)).data.T
        fpfh_22 = o3d.pipelines.registration.compute_fpfh_feature(surf_2,
                  o3d.geometry.KDTreeSearchParamRadius(radius=10)).data.T
        fpfh_23 = o3d.pipelines.registration.compute_fpfh_feature(surf_2,
                  o3d.geometry.KDTreeSearchParamRadius(radius=20)).data.T
        
        idx_1, idx_2 = self.feature_indices(surface_1=surf_1, surface_2=surf_2, anchor_1=anchor_1, anchor_2=anchor_2)

        combined_1, combined_2 = [], []
        for i in range(len(idx_1)):
            feat_1 = np.hstack((pts_R[idx_1[i],:], normals_R[idx_1[i],:], fpfh_11[idx_1[i],:],
                                fpfh_12[idx_1[i],:], fpfh_13[idx_1[i],:]))
            feat_2 = np.hstack((pts_L[idx_2[i],:], normals_L[idx_2[i],:], fpfh_21[idx_2[i],:],
                                fpfh_22[idx_2[i],:], fpfh_23[idx_2[i],:]))
            combined_1.append(feat_1)
            combined_2.append(feat_2)

        combined_1 = np.asarray(combined_1).reshape((-1, 105))
        combined_2 = np.asarray(combined_2).reshape((-1, 105))

        return np.vstack((combined_1, combined_2))

    @staticmethod
    def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Iterative furthest point sampling to select a set of n point features 
        that have the largest minimum distance.

        Parameters
        ----------
        xyz : BxNx3 : obj : 'torch.Tensor'
            pointcloud data
        npoint : int
            number of samples
        
        Returns
        -------
        centroids : BxM : obj : 'torch.Tensor' 
            sampled pointcloud index 
        """
        device = xyz.device
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]

        return centroids

    @staticmethod
    def feature_indices(surface_1, surface_2, anchor_1: list, anchor_2: list) -> list:
        """
        Select point indices of contact surfaces that will be used for feature selection.

        Parameters
        ----------
        surface_1 : obj : `open3d.geometry.PointCloud`
            point cloud of a contact surface 1
        surface_2 : obj : `open3d.geometry.PointCloud`
            point cloud of a contact surface 2
        anchor_1 : 1xN : obj : `list`
            indices of anchor points on a surface 1
        anchor_2 : 1xN : obj : `list` 
            indices of anchor points on a surface 2
        
        Returns
        -------
        feat_idx1 : 1xN : obj : `list` 
            selected indices on a contact surface 1
        feat_idx2 : 1xN : obj : `list` 
            selected indices on a contact surface 2
        """
        feat_idx1, feat_idx2 = [], []        
        surf1_tree = o3d.geometry.KDTreeFlann(surface_1)
        surf2_tree = o3d.geometry.KDTreeFlann(surface_2)

        for i in range(len(anchor_1)):
            [_, idx_1, _] = surf1_tree.search_knn_vector_3d(surface_1.points[anchor_1[i]], 20)
            [_, idx_2, _] = surf2_tree.search_knn_vector_3d(surface_2.points[anchor_2[i]], 20)
            feat_idx1.append(idx_1)
            feat_idx2.append(idx_2)

        return feat_idx1, feat_idx2
    
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.llg = LLGBlock(dim=3, embed_dim=105)
        self.mixer = SimpleMixer(dim=105, num_patch=807, embed_dim=256, depth=4, token_dim=128, channel_dim=1024,
                                 dropout=0.1, num_classes=1)

    def forward(self, x):
        proj = self.llg(x[1])
        x = torch.cat((x[0], proj), 1)
        x = self.mixer(x)

        return x
