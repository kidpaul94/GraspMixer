import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader

from engine import Engine
from models import MyModel
from utils import Val_Dataset

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='CPPE')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use CUDA to evaluate model.')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='name of pretrained weights, if exists.')
    parser.add_argument('--dataset_path', default='../dataset/test', type=str,
                        help='path to evaluation dataset.')
    parser.add_argument('--csv_file', default='summary.csv', type=str,
                        help='summary file of training and validation dataset.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size to train the NNs.')

    global args
    args = parser.parse_args(argv)

def eval(args) -> None:
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f'Use {device} for evaluation...')
    device = torch.device(device)

    dataset = Val_Dataset(root_dir=args.dataset_path, csv_file=args.csv_file)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    if args.pretrained is not None:
        checkpoint = torch.load(f'./weights/{args.pretrained}.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Load model from {args.pretrained}.pth')
    else:
        print('Pretrained weights not provided!!!')
        return

    criterion = nn.BCEWithLogitsLoss()
    trainer = Engine(model=model, loaders=eval_loader, criterion=criterion, device=device)  

    _, acc, prec, rec, f1 = trainer.validate()    

    print('------------------------------------------------')
    print(f'Acc: {acc:0.4f} | Prec: {prec:0.4f} | Rec: {rec:0.4f} | F1 score: {f1:0.4f}')
    print('------------------------------------------------')

    print('Finished evaluation!!!')    

if __name__ == "__main__":
    parse_args()
    eval(args)
