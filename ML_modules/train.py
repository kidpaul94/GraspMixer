import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import transforms as T
from engine import Engine
from models import MyModel
from utils import Simple_Dataset

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='PTAE')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use CUDA to train model.')
    parser.add_argument('--logging', default='./logs', type=str,
                        help='path to save a log file.')
    parser.add_argument('--pretrained', default='weights_2', type=str,
                        help='name of pretrained weights, if exists.')
    parser.add_argument('--dataset_path', default='../dataset/train', type=str,
                        help='path to training dataset.')
    parser.add_argument('--csv_file', default='summary.csv', type=str,
                        help='summary file of training dataset.')
    parser.add_argument('--save_path', default='./weights', type=str,
                        help='Directory for saving checkpoint models.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size to train the NNs.')
    parser.add_argument('--num_epochs', default=200, type=int,
                        help='# of epoch to train the NNs.')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate to train.')
    parser.add_argument('--step_size', default=5, type=int,
                        help='lr step size')
    parser.add_argument('--gamma', default=0.3, type=float,
                        help='For each lr step, what to multiply the lr by')

    global args
    args = parser.parse_args(argv)

def train(args) -> None:
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f'Use {device} for training...')
    device = torch.device(device)

    augmentation = T.Compose([T.ToTensor()]) # T.RandomJitter(), 
    dataset = Simple_Dataset(root_dir=args.dataset_path, csv_file=args.csv_file, 
                             transform=augmentation)

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    start, min_loss = 0, 1e4

    if args.pretrained is not None:
        checkpoint = torch.load(f'./weights/{args.pretrained}.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start, min_loss = checkpoint['epoch'], checkpoint['loss']
        print(f'Load the checkpoint from {args.pretrained}.pth')
    
    trainer = Engine(model=model, loaders=[train_loader, val_loader], 
                     criterion=criterion, device=device)
    writer = SummaryWriter(log_dir=args.logging)

    for i in range(start, args.num_epochs):
        train_loss, val_loss = trainer.train_one_epoch(optim=optimizer, epoch=i, 
                                                        scheduler=scheduler)
        min_loss = trainer.snapshot(min_loss=min_loss, loss=val_loss, save_dir=args.save_path,
                                    epoch=i, optim=optimizer, scheduler=scheduler)

        if args.logging is not None:
            print('Save current loss values to Tensorboard...')
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train Loss', train_loss, i)
            writer.add_scalar('Eval Loss', val_loss, i)
            writer.add_scalar('Learning Rate', lr, i)          

    print('Finished training!!!')    

if __name__ == "__main__":
    parse_args()
    train(args)
