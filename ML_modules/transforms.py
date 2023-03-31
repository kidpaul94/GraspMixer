import torch
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, features):
        """ 
        Combine multiple augmentation processes and apply them sequentially.
        
        Parameters
        ----------
        features : Nx3
            initial input features
            
        Returns
        -------
        features : Nx3 
            transformed features
        """
        for t in self.transforms:
            features = t(features)

        return features

class ToTensor(object):

    @staticmethod
    def __call__(features):
        """ 
        Convert np.ndarray features to torch.Tensor features.
        
        Parameters
        ----------
        features : Nx3 : obj : np.ndarray
            initial input features
            
        Returns
        -------
        features : Nx3 : obj : torch.Tensor
            converted torch.Tensor features
        """
        features = torch.from_numpy(features)
        if not isinstance(features, torch.FloatTensor):
            features = features.float()

        return features

class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1]):
        self.angle = angle

    def __call__(self, features):
        """ 
        Randomly rotate an input.
        
        Parameters
        ----------
        angle : 1X3 : obj : `list`
            list of paramters to control random rotation in each axis 
        features : Nx3 : obj : np.ndarray
            initial input features
            
        Returns
        -------
        features : Nx3 : obj : np.ndarray
            transformed features
        """
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        features[:,:3] = np.dot(features[:,:3], np.transpose(R))

        return features
    
class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
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
        features : Nx3 : obj : np.ndarray
            initial input features
            
        Returns
        -------
        features : Nx3 : obj : np.ndarray
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
        features : Nx3 : obj : np.ndarray
            initial input features
            
        Returns
        -------
        features : Nx3 : obj : np.ndarray
            transformed features
        """
        features = features[np.random.permutation(features.shape[0]), :]

        return features

class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, features):
        """ 
        Add jitters to the given point cloud features.
        
        Parameters
        ----------
        sigma : float
            standard deviation of a sampling distribution
        clip : float
            cliping value of randomly generated jitters
        features : Nx3 : obj : np.ndarray
            initial input features
            
        Returns
        -------
        features : Nx3 : obj : np.ndarray
            features with jitters
        """
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(features.shape[0], 3), -1 * self.clip, self.clip)
        features[:,:3] += jitter
        
        return features