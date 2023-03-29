import os
import copy
import torch
import numpy as np
import pandas as pd
import open3d as o3d
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
            data_misc = self.transform(data_misc)

        label = class_type(label=label, num_class=1)

        return (data_pts, data_misc, label, data_path)

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

            for i in range(0, len(sort_paths) - 1, 2):
                if cpps[i // 2] >= 0.0:
                    item = f'{folder.name}/{sort_paths[i][:4]}' 
                    data.append(item)
                    prob.append(cpps[i // 2])
        else:
            print(f'Not a directory: {folder.name}')

    list_dict = {'Data': data, 'Prob': prob} 
    df = pd.DataFrame(list_dict) 
    df.to_csv(f'{root_dir}/summary.csv', index=False)

def class_type(label: float, num_class: int = 10):
    """ 
    Adjust label format depending on a classification type.
    
    Parameters
    ----------
    label : float
        original label
    num_class : int
        number of classes
        
    Returns
    -------
    label : converted label
    """
    if num_class > 1:
        temp = int(round(label, 1) * 10) - 1
        label = torch.LongTensor(temp)
    else:
        print('# of classes has to be larger than 1!!!')

    return label

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
