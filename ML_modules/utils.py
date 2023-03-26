import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Simple_Dataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform = None) -> None:
        self.root_dir = root_dir
        self.annotations = pd.read_csv(f'{root_dir}/{csv_file}')
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx, 0]
        data_path = os.path.join(self.root_dir, item)
        label = self.annotations.iloc[idx, 1]
        data_pts = np.load(f'{data_path}_pts.npy')
        data_misc = np.load(f'{data_path}_misc.npy')

        if self.transform is not None:
            data_pts = self.transform(data_pts)

        return (data_pts, data_misc, label)

def gen_csv(root_dir: str) -> None:
    """ 
    Generate summary of the dataset in .csv format.
    
    Parameters
    ----------
    root_dir : str
        root directory of the dataset
        
    Returns
    -------
    None
    """
    data, prob = [], []
    for folder in os.scandir(root_dir):
        if folder.is_dir():
            sort_paths = sorted(os.listdir(folder.path))
            with open(f'{folder.path}/prob.txt') as f:
                cpps = eval(f.read())
            prob.extend(cpps)

            for i in range(0, len(sort_paths) - 1, 2):
                item = f'{folder.name}/{sort_paths[i][:4]}'
                data.append(item)
        else:
            print(f'Not a directory: {folder.name}')

    list_dict = {'Data': data, 'Prob': prob} 
    df = pd.DataFrame(list_dict) 
    df.to_csv(f'{root_dir}/summary.csv', index=False)
