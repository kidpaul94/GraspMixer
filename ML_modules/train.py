import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import transforms as T
from engine import Engine
from models import MyModel
from utils import Train_Dataset

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='CPPE')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use CUDA to train model.')
    parser.add_argument('--logging', default='./logs', type=str,
                        help='path to save a log file.')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='name of pretrained weights, if exists.')
    parser.add_argument('--train_path', default='../dataset/train', type=str,
                        help='path to training dataset.')
    parser.add_argument('--val_path', default='../dataset/val', type=str,
                        help='path to validation dataset.')
    parser.add_argument('--csv_file', default='summary.csv', type=str,
                        help='summary file of training and validation dataset.')
    parser.add_argument('--save_path', default='./weights', type=str,
                        help='Directory for saving checkpoint models.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size to train the NNs.')
    parser.add_argument('--num_epochs', default=60, type=int,
                        help='# of epoch to train the NNs.')
    parser.add_argument('--lr', default=[1e-6, 5e-4], type=float, nargs='+',
                        help='minimum and maximum learning rate to train.')
    parser.add_argument('--step', default=[10, 20], type=int, nargs='+',
                        help='warmup_steps and first_cycle_steps.')
    parser.add_argument('--gamma', default=0.7, type=float,
                        help='For each lr step, what to multiply the lr by')

    global args
    args = parser.parse_args(argv)

def data_balance(dataset, class_weights: list = [63.8, 36.2]):
    """ 
    Generate a sampler to draw data with given class probabilities.
    
    Parameters
    ----------
    dataset : obj : 'torch.utils.data.dataset.Subset'
        dataset that will use a sampler
    class_weights : 1X2 : obj : `list` 
        class weights of two classes
        
    Returns
    -------
    sampler : obj : 'torch.utils.data.WeightedRandomSampler'
        weighted random sampler
    """
    sample_weights = [0] * len(dataset)
    for idx, (_, _, label, _) in enumerate(dataset):
        sample_weights[idx] = class_weights[int(label.item())]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

    return sampler

def train(args) -> None:
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f'Use {device} for training...')
    device = torch.device(device)

    augmentation = [T.Compose([T.RandomRotate(), T.RandomPermute(), T.RandomScale()]), 
                    T.RandomJitter(is_pts=False)]
    train_set = Train_Dataset(root_dir=args.train_path, csv_file=args.csv_file, transform=augmentation)       
    val_set = Train_Dataset(root_dir=args.val_path, csv_file=args.csv_file)
    # dataset = Train_Dataset(root_dir=args.train_path, csv_file=args.csv_file, 
    #                          transform=augmentation)
    # train_size = int(len(dataset) * 0.9)
    # valid_size = len(dataset) - train_size
    # train_set, val_set = random_split(dataset, [train_size, valid_size])

    sampler = data_balance(dataset=train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr[1], momentum=0.9)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.step[1], max_lr=args.lr[1],  
                                              min_lr=args.lr[0], warmup_steps=args.step[0], gamma=args.gamma)
    
    start = 0
    if args.pretrained is not None:
        checkpoint = torch.load(f'./weights/{args.pretrained}.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start = checkpoint['epoch']
        print(f'Load the checkpoint from {args.pretrained}.pth')
    
    trainer = Engine(model=model, loaders=[train_loader, val_loader], 
                     criterion=criterion, device=device)
    writer = SummaryWriter(log_dir=args.logging)

    for i in range(start, args.num_epochs):
        train_loss, val_loss, acc, prec, rec, f1 = trainer.train_one_epoch(optim=optimizer, epoch=i, 
                                                                scheduler=scheduler)
        trainer.snapshot(save_dir=args.save_path, epoch=i, optim=optimizer, scheduler=scheduler)

        if args.logging is not None:
            print('Save current training records to Tensorboard...')
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalars('Loss Logging', {'Train': train_loss, 'Val': val_loss}, i)
            writer.add_scalars('Score', {'Accunracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}, i)
            writer.add_scalar('Learning Rate', lr, i)          

    print('Finished training!!!')    

if __name__ == "__main__":
    parse_args()
    train(args)
