import os
import time
import torch
import numpy as np
from tqdm import tqdm

from utils import save_output

class Engine(object):
    def __init__(self, model, loaders, criterion, device: str = 'cuda'):
        self.model, self.device = model.to(device), device
        self.loaders, self.criterion = loaders, criterion

    def train_one_epoch(self, optim, epoch: int, scheduler = None) -> tuple:
        """ 
        Train NNs model only one epoch.
        
        Parameters
        ----------
        optim : string
            optimization function used for the training process
        epoch : int
            current epoch number (iteration)
        scheduler : obj : torch.optim.lr_scheduler
            learning rate scheduler used for the training process
            
        Returns
        -------
        tuple : average training and validation losses
        """
        self.model.train()
        print(f'Start #{epoch+1} epoch...')
        epoch_start_time = time.time()
        loss_buf = []

        for _, data in tqdm(enumerate(self.loaders[0]), total=len(self.loaders[0])):
            feature_1, feature_2, label, _ = data
            feature_1 = feature_1.to(self.device)
            feature_2 = feature_2.to(self.device)
            
            optim.zero_grad()
            pred = self.model([feature_1, feature_2])
            loss = self.criterion(pred, label)
            loss.backward()
            optim.step()
            loss_buf.append(loss.detach().cpu().numpy())

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            val_loss = self.validate()

        train_loss = np.mean(loss_buf)
        epoch_time = time.time() - epoch_start_time
        print(f'Train_loss: {train_loss} | Val_loss: {val_loss} | Epoch_time: {epoch_time:0.4f}s')

        return (train_loss, val_loss)

    @torch.inference_mode()
    def validate(self, show_res: bool = False) -> float:
        """ 
        Validate the trained model using the validation dataset.
        
        Parameters
        ----------
        show_res : bool
            whether visualize predictions or not during the process
            
        Returns
        -------
        val_loss : float
            whether put a point cloud's frame to its geometric centroid
        """
        self.model.eval()
        loader = self.loaders[1] if len(self.loaders) == 2 else self.loaders
        print('Start valuation...')
        loss_buf = []

        for _, data in tqdm(enumerate(loader), total=len(loader)):
            feature_1, feature_2, label, path = data
            feature_1 = feature_1.to(self.device)
            feature_2 = feature_2.to(self.device)

            pred = self.model([feature_1, feature_2])
            loss = self.criterion(pred, label)
            loss_buf.append(loss.detach().cpu().numpy())

        val_loss = np.mean(loss_buf)

        if show_res:
            for i in range(len(label)):
                save_output(save_dir='./results', data_name=path[i], prob=label[i])

        return val_loss

    def snapshot(self, epoch: int, min_loss: float, loss: float, save_dir: str) -> float:
        """ 
        Save the trained model-weights.
        
        Parameters
        ----------
        epoch : int
            current epoch number (iteration)
        min_loss : float
            minimum loss value that has been saved
        loss : float
            calculated loss value of the current iteration
        save_dir : str
            saving directory of the current model weights
            
        Returns
        -------
        min_loss : float
            updated min_loss value
        """
        if not os.path.exists(save_dir):
            print(f'Generate weights folder...')
            os.mkdir(save_dir)

        if loss < min_loss:
            min_loss = loss
            print(f'Saving #{epoch} epoch model weights to {save_dir}...')
            torch.save(self.model.state_dict(), f'{save_dir}/weights_{epoch}.pth')
            
        return min_loss
