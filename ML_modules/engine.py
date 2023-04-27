import os
import time
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_recall, binary_f1_score

from utils import save_output

class Engine(object):
    def __init__(self, model, loaders, criterion, device: str = 'cuda'):
        self.model, self.device = model, device
        self.loaders, self.criterion = loaders, criterion

    def train_one_epoch(self, optim, epoch: int, scheduler = None) -> tuple:
        """ 
        Train NNs model only one epoch.
        
        Parameters
        ----------
        optim : obj : 'torch.optim'
            optimization function used for the training process
        epoch : int
            current epoch number (iteration)
        scheduler : obj : 'torch.optim.lr_scheduler'
            learning rate scheduler used for the training process
            
        Returns
        -------
        tuple : average training & validation losses and accunracy & f1-score
        """
        self.model.train()
        print(f'Start #{epoch+1} epoch...')
        epoch_start_time = time.time()
        loss_buf, pred_buf, label_buf = [], [], []

        for _, data in tqdm(enumerate(self.loaders[0]), total=len(self.loaders[0])):
            feature_1, feature_2, label, _ = data 
            feature_1 = feature_1.to(self.device)
            feature_2 = feature_2.to(self.device)

            label_buf.append(label)
            label = label.to(self.device)

            optim.zero_grad()
            pred = self.model([feature_1, feature_2])
            loss = self.criterion(pred, label)
            loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            optim.step()
            loss_buf.append(loss.item())
            pred_buf.append(pred.detach().cpu())
    
        label_buf = torch.cat(label_buf).squeeze_()
        pred_buf = torch.sigmoid(torch.cat(pred_buf).squeeze_())
        acc_t = binary_accuracy(preds=pred_buf, target=label_buf)
        prec_t = binary_precision(preds=pred_buf, target=label_buf)
        rec_t = binary_recall(preds=pred_buf, target=label_buf)
        f1_t = binary_f1_score(preds=pred_buf, target=label_buf)

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            val_loss, acc, prec, rec, f1 = self.validate()

        train_loss = sum(loss_buf) / len(loss_buf)
        epoch_time = time.time() - epoch_start_time
        print('------------------------------------------------')
        print(f'Epoch_time: {epoch_time:0.4f}s')
        print(f'Train_loss: {train_loss:0.4f} | Val_loss: {val_loss:0.4f}')
        print('------------------------------------------------')
        print('Training:')
        print(f'Acc: {acc_t:0.4f} | Prec: {prec_t:0.4f} | Rec: {rec_t:0.4f} | F1 score: {f1_t:0.4f}')
        print('------------------------------------------------')
        print('Validation:')
        print(f'Acc: {acc:0.4f} | Prec: {prec:0.4f} | Rec: {rec:0.4f} | F1 score: {f1:0.4f}')
        print('------------------------------------------------')

        return (train_loss, val_loss, acc, prec, rec, f1)

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
        acc : float
            accuracy value of validation set
        prec : float
            precision value of validation set
        recall : float
            recall value of validation set
        f1 : float
            f1-score value of validation set
        """
        self.model.eval()
        loader = self.loaders[1] if len(self.loaders) == 2 else self.loaders
        print('Start valuation...')
        loss_buf, pred_buf, label_buf = [], [], []

        for _, data in tqdm(enumerate(loader), total=len(loader)):
            feature_1, feature_2, label, path = data  
            feature_1 = feature_1.to(self.device)
            feature_2 = feature_2.to(self.device)

            label_buf.append(label)
            label = label.to(self.device)

            pred = self.model([feature_1, feature_2]) 
            loss = self.criterion(pred, label)
            loss_buf.append(loss.item())
            pred_buf.append(pred.detach().cpu())

        label_buf = torch.cat(label_buf).squeeze_()
        pred_buf = torch.sigmoid(torch.cat(pred_buf).squeeze_())
        acc = binary_accuracy(preds=pred_buf, target=label_buf)
        prec = binary_precision(preds=pred_buf, target=label_buf)
        rec = binary_recall(preds=pred_buf, target=label_buf)
        f1 = binary_f1_score(preds=pred_buf, target=label_buf)

        val_loss = sum(loss_buf) / len(loss_buf)

        if show_res:
            for i in range(len(label)):
                save_output(save_dir='./results', data_name=path[i], prob=label[i])
                
        return val_loss, acc, prec, rec, f1

    def snapshot(self, save_dir: str, epoch: int, optim, scheduler = None) -> float:
        """ 
        Save the trained model-checkpoint.
        
        Parameters
        ----------
        save_dir : str
            saving directory of the current model checkpoint
        epoch : int
            current epoch number (iteration)
        optim : obj : 'torch.optim'
            optimization function used for the training process
        scheduler : obj : 'torch.optim.lr_scheduler'
            learning rate scheduler used for the training process
            
        Returns
        -------
        None
        """
        if not os.path.exists(save_dir):
            print(f'Generate weights folder...')
            os.mkdir(save_dir)

        print(f'Saving #{epoch + 1} epoch model checkpoint to {save_dir}...')
        dict = {'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1}
        torch.save(dict, f'{save_dir}/checkpoint_{epoch + 1}.pth')
