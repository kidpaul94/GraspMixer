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
        label = item.split('/', 1)[0]
        data = np.load(data_path)

        if self.transform is not None:
            data = self.transform(data)

        return (data, label)
