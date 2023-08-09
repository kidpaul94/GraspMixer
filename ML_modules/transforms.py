import os
import sys
import inspect
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from gripper_config import params

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, features):
        """ 
        Combine multiple augmentation processes and apply them sequentially.
        
        Parameters
        ----------
        features : Nx6
            initial input features
            
        Returns
        -------
        features : Nx6 
            transformed features
        """
        for t in self.transforms:
            features = t(features)

        return features

class ToTensor(object):

    @staticmethod
    def __call__(features):
        """ 
        Convert numpy.ndarray features to torch.Tensor features.
        
        Parameters
        ----------
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'torch.Tensor'
            converted torch.Tensor features
        """
        features = torch.from_numpy(features)
        if not isinstance(features, torch.FloatTensor):
            features = features.float()

        return features

class RandomRotate(object):
    def __init__(self, angle: list = [1, 1, 1]):
        self.angle = angle

    def __call__(self, features):
        """ 
        Randomly rotate an input.
        
        Parameters
        ----------
        angle : 1X3 : obj : `list`
            list of paramters to control random rotation in each axis 
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            transformed features
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(features[:,:3])
        pcd.normals = o3d.utility.Vector3dVector(features[:,3:])

        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        RM = R.from_euler('xyz', [angle_x, angle_y, angle_z]).as_matrix()
        pcd.rotate(R=RM, center=(0, 0, 0))

        features[:,:3] = np.asarray(pcd.points)
        features[:,3:] = np.asarray(pcd.normals)

        return features
    
class RandomScale(object):
    def __init__(self, scale: list = [0.97, 1.03], anisotropic: bool = False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, features):
        """ 
        Randomly scale an input.
        
        Parameters
        ----------
        scale : 1X2 : obj : `list`
            min & max  scaling factors
        anisotropic : bool
            whether equally scale an input in every dimension
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            transformed features
        """
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        features[:,:3] *= scale

        return features
    
class RandomPermute(object):

    @staticmethod
    def __call__(features):
        """ 
        Randomly rotate an input.
        
        Parameters
        ----------
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            transformed features
        """
        features = features[np.random.permutation(features.shape[0]), :]

        return features

class RandomJitter(object):
    def __init__(self, sigma: list = [0.04], clip: list = [0.07], is_pts: bool = True):
        self.sigma, self.clip = sigma, clip
        self.is_pts = is_pts

    def __call__(self, features):
        """ 
        Add jitters to the given point cloud features.
        
        Parameters
        ----------
        sigma : 1xN : obj : 'list'
            standard deviation of a sampling distribution
        clip : 1xN : obj : 'list'
            cliping value of randomly generated jitters
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            features with jitters
        """
        if self.is_pts:
            assert (self.clip[0] > 0)
            jitter = np.clip(self.sigma[0] * np.random.randn(features.shape[0], 3), -1 * self.clip[0], self.clip[0])
            features[:,:3] += jitter
        else:
            assert (len(self.clip[0]) == 5)
            for i in range(3):
                jitter = np.clip(self.sigma[i] * np.random.randn(features.shape[0], 1), -1 * self.clip[i], self.clip[i])
                features[0,i] += jitter

            for j in range(2):
                multi = np.clip(self.sigma[3] * np.random.randn(features.shape[0], 1), -1 * self.clip[3], self.clip[3])
                jitter = np.random.rand(3)
                features[j+4,:3] = params['gripper_force'] * features[j+4,:3] + multi * jitter / np.linalg.norm(jitter)
                features[j+4,:3] = features[j+4,:3] / np.linalg.norm(features[j+4,:3])
  
            jitter = np.clip(self.sigma[4] * np.random.randn(features.shape[0], 3), -1 * self.clip[4], self.clip[4])
            features[6,:3] += jitter        

        return features
