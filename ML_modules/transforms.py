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
            np.ndarray input features
            
        Returns
        -------
        features : Nx3 : obj : torch.Tensor
            converted torch.Tensor features
        """
        features = torch.from_numpy(features)
        if not isinstance(features, torch.FloatTensor):
            features = features.float()
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
        features : Nx6 : obj : np.ndarray
            input features
            
        Returns
        -------
        features : Nx3 : obj : np.ndarray
            features with jitters
        """
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(features.shape[0], 3), -1 * self.clip, self.clip)
        features[:,:] += jitter
        return features