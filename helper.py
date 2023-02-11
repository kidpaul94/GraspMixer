import random
import argparse
import numpy as np
import open3d as o3d

from CPPE_utils import gen_pointcloud, Visualization

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='CPPE')
    parser.add_argument('--choose_task', default='generate', type=str,
                        help='choose a functionality of the code: generate or visualize.')
    parser.add_argument('--object_path', default='./objects/pcds/4096_obj_05.pcd', type=str,
                        help='path of an object we wish to evaluate.')
    parser.add_argument('--num_pts', default=4096, type=int,
                        help='number of points in the .pcd file.')
    parser.add_argument('--dict_path', default='./objects/dicts/4096_obj_05/4096_obj_05.txt', type=str,
                        help='path of a grasp dictionary.')
    parser.add_argument('--idx', default=None, type=int,
                        help='number of visualized grasps in a grasp dictionary')

    global args
    args = parser.parse_args(argv)

def visualize_result(path: str, path_dict: str = None, idx: int = None) -> None:
    pcd = o3d.io.read_point_cloud(path)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    list2vis = [pcd, frame]
    vis = Visualization(pcd=pcd)
    
    if idx is not None:
        to_add = []
        with open(path_dict) as f:
            cpps = eval(f.read())
        cpps = np.asarray(cpps)
        chosen = random.sample(range(len(cpps)), idx)
        for i in chosen:
            contact_p = cpps[i].reshape(-1,3)
            gripper = vis.single_cpp(contact_p=contact_p)
            to_add = to_add + gripper
        print(chosen)
    else:
        to_add = vis.all_centers(path_dict=path_dict)
    list2vis = list2vis + to_add

    o3d.visualization.draw_geometries(list2vis, point_show_normal=False)

if __name__ == "__main__":
    parse_args()
    if args.choose_task == 'generate':
        gen_pointcloud(path=args.object_path, num_points=args.num_pts)
    elif args.choose_task == 'visualize': 
        visualize_result(path=args.object_path, path_dict=args.dict_path, idx=args.idx)
