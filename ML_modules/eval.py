import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader

import transforms as T
from engine import Engine
from models import MyModel
from utils import Simple_Dataset

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='PTAE')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use CUDA to evaluate model.')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='name of pretrained weights, if exists.')
    parser.add_argument('--dataset_path', default='./dataset', type=str,
                        help='path to evaluation dataset.')
    parser.add_argument('--save_path', default='./weights', type=str,
                        help='Directory for saving evaluation results.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size to train the NNs.')
    parser.add_argument('--input_size', default=256, type=int,
                        help='input_size to match dimension of training data.')

    global args
    args = parser.parse_args(argv)

def eval(args) -> None:
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f'Use {device} for evaluation...')
    device = torch.device(device)

    augmentation = T.Compose([T.ToTensor()])
    dataset = Simple_Dataset(root_dir=args.dataset_path, csv_file=args.csv_file, 
                             transform=augmentation)

    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel()
    if args.pretrained is not None:
        state_dict = torch.load(f'./weights/{args.pretrained}.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        print(f'Load model from {args.pretrained}.pth')
    else:
        print('Pretrained weights not provided!!!')
        return

    criterion = nn.BCEWithLogitsLoss()
    trainer = Engine(model=model, loaders=eval_loader, criterion=criterion, device=device)  

    for i in range(args.epoch):
        _ = trainer.validate(epoch=i, show_res=True, save_dir=args.save_path)        

    print('Finished evaluation!!!')    

if __name__ == "__main__":
    parse_args()