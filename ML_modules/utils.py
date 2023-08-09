import os
import copy
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from torch.utils.data import Dataset

import transforms as T
from models import MSG_fpfh
from quality import GraspMetrics

class Train_Dataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform = None) -> None:
        self.root_dir = root_dir
        self.annotations = pd.read_csv(f'{root_dir}/{csv_file}')
        self.transform, self.toTensor = transform, T.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx, 0]
        data_path = os.path.join(self.root_dir, item)
        label = self.annotations.iloc[idx, 1]
        cld_R = np.load(f'{data_path}R_pts.npy')
        cld_L = np.load(f'{data_path}L_pts.npy')
        data_pts, pin = np.vstack((cld_R, cld_L)), len(cld_R)

        misc_1 = np.load(f'{data_path}_misc_1.npy')
        misc_2 = np.load(f'{data_path}_misc_2.npy', allow_pickle=True)

        if self.transform is not None:
            data_pts = self.transform[0](data_pts)
            misc_1 = self.transform[1](misc_1)
        metric = GraspMetrics(misc_1[1], misc_1[2:4], misc_1[4:6], misc_1[7:], misc_1[6]) 
        data_misc = metric.Q_combined([misc_1[0,0], misc_1[0,0]], misc_2, 
                                      misc_1[0,1], misc_1[0,2])
        
        SR, SL = data_pts[:pin,:3], data_pts[pin:,:3]
        NR, NL = data_pts[:pin,3:], data_pts[pin:,3:]
        data_pts = MSG_fpfh().gen_features(pts_R=SR, pts_L=SL, normals_R=NR, normals_L=NL)
        data_pts, data_misc = self.toTensor(data_pts), self.toTensor(data_misc)

        label = class_type(label=label, num_class=1)

        return (data_pts, data_misc, label, data_path)
    
class Val_Dataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str) -> None:
        self.root_dir = root_dir
        self.annotations = pd.read_csv(f'{root_dir}/{csv_file}')
        self.toTensor = T.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx, 0]
        data_path = os.path.join(self.root_dir, item)
        label = self.annotations.iloc[idx, 1]
        data_pts = np.load(f'{data_path}_pts.npy')
        misc_1 = np.load(f'{data_path}_misc_1.npy')
        misc_2 = np.load(f'{data_path}_misc_2.npy', allow_pickle=True)
        metric = GraspMetrics(misc_1[1], misc_1[2:4], misc_1[4:6], misc_1[7:], misc_1[6])  
        data_misc = metric.Q_combined([misc_1[0,0], misc_1[0,0]], misc_2, 
                                      misc_1[0,1], misc_1[0,2])

        data_pts = self.toTensor(data_pts)
        data_misc = self.toTensor(data_misc)

        label = class_type(label=label, num_class=1)

        return (data_pts, data_misc, label, data_path)

def gen_csv(root_dir: str, is_training: bool = True) -> None:
    """ 
    Generate summary of the dataset in .csv format.
    
    Parameters
    ----------
    root_dir : str
        root directory of the dataset
    is_training : bool
        whether generate a .csv for training or validation set
        
    Returns
    -------
    None
    """
    data, prob = [], []
    space = 3 if is_training else 2
    for folder in os.scandir(root_dir):
        if folder.is_dir():
            sort_paths = sorted(os.listdir(folder.path))
            with open(f'{folder.path}/prob.txt') as f:
                cpps = eval(f.read())

            for i in range(0, len(sort_paths) - 1, space):
                if cpps[i // space] >= 0.0:
                    item = f'{folder.name}/{sort_paths[i][:4]}' 
                    data.append(item)
                    prob.append(cpps[i // space])
        else:
            print(f'Not a directory: {folder.name}')

    list_dict = {'Data': data, 'Prob': prob} 
    df = pd.DataFrame(list_dict) 
    df.to_csv(f'{root_dir}/summary.csv', index=False)

def class_type(label: float, num_class: int = 3, threshold: float = 0.85):
    """ 
    Adjust label format depending on a classification type.
    
    Parameters
    ----------
    label : float
        original label
    num_class : int
        number of classes
    threshold : float
        threshold value for binary classification
        
    Returns
    -------
    label : converted label
    """
    assert num_class > 0, '# of classes has to be larger than 1'
    if num_class > 1:
        if label < 0.4:
            temp = 0 # Futile
        elif label >= 0.6:
            temp = 2 # Robust
        else:
            temp = 1 # Fragile
        label = torch.LongTensor([temp])
    else:
        label = 1.0 if label > threshold else 0.0
        label = torch.FloatTensor([label])

    return label

def model_size(model: torch.nn.Module) -> int:
    """ 
    Calculate the total # of parameters in the model.
    
    Parameters
    ----------
    model : obj : 'torch.nn.Module' 
        neural network model to check
        
    Returns
    -------
    pytorch_total_params : int
        total # of parameters
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    return pytorch_total_params

def save_output(save_dir: str, data_name: str, prob: float) -> None:
    """ 
    Save image of contact surface on an object and its associated 
    success probability.
    
    Parameters
    ----------
    save_dir : str
        saving directory of the visualizations
    data_name : str
        name of the input contact surface
    prob : float
        probability of success
        
    Returns
    -------
    None
    """
    if not os.path.exists(save_dir):
        print(f'Generate results folder...')
        os.mkdir(save_dir)

    obj_name = data_name[:-5]
    data_pts = np.load(f'../dataset/train/{data_name}_pts.npy')
    grasp = o3d.geometry.PointCloud()
    grasp.points = o3d.utility.Vector3dVector(data_pts[:,:3])
    grasp.paint_uniform_color([0, 1, 0])
    pcd = o3d.io.read_point_cloud(f'../objects/pcds/{obj_name}.pcd')

    idx = int(data_name[-5:])
    with open(f'../objects/dicts/{obj_name}/{obj_name}_cpps.txt') as f:
        cpp = eval(f.read())[idx]

    mesh_1 = o3d.geometry.TriangleMesh.create_sphere(radius=5, resolution=5).paint_uniform_color([1, 0.7, 0])
    mesh_2 = copy.deepcopy(mesh_1)
    mesh_1.translate((cpp[0], cpp[1], cpp[2]), relative=False)
    mesh_2.translate((cpp[3], cpp[4], cpp[5]), relative=False)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    for data in [pcd, grasp, mesh_1, mesh_2]:
        vis.add_geometry(data)
        vis.update_geometry(data)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename=f'{save_dir}/{data_name}_{prob:04f}.png', do_render=False)
    vis.destroy_window()
