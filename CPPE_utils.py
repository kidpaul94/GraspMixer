import os
import copy
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from kmedoids import KMedoids
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import euclidean_distances

from gripper_config import params

def gen_pointcloud(path: str, num_points: int = 4096, zero_mean: bool = False,
                   visualize: bool = False) -> None:
    """ 
    Generate a point cloud from a mesh file.
    
    Parameters
    ----------
    path : string
        path to a mesh file
    num_points : int
        number of points in a point cloud
    zero_mean : bool
        whether put a point cloud's frame to its geometric centroid
    visualize : bool
        whether visualize a generated point cloud
        
    Returns
    -------
    None
    """
    mesh = o3d.io.read_triangle_mesh(path)
    CAD = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)
    pts_CAD = np.asarray(CAD.points)

    if CAD.normals is None:
        print('Estimate normal vectors...')
        CAD.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=10))
        CAD.orient_normals_consistent_tangent_plane(k=10)
    else:
        normals_CAD = np.asarray(CAD.normals)

    pcd = o3d.geometry.PointCloud()
    if zero_mean:
        print('Move CAD coordinate frame to its geometric centroid...')
        temp = pts_CAD - pts_CAD.mean(axis=0, keepdims=True)
        pcd.points = o3d.utility.Vector3dVector(temp)
    else:
        pcd.points = o3d.utility.Vector3dVector(pts_CAD)
    pcd.normals = o3d.utility.Vector3dVector(normals_CAD)    
    pcd.normalize_normals()
    pcd.paint_uniform_color([1, 0, 1])

    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    file = os.path.basename(path)
    filename, _ = os.path.splitext(file)
    o3d.io.write_point_cloud(f'./objects/{num_points}_{filename}.pcd', pcd)

def rot_matrix(p_axis: np.ndarray, norm: np.float64, d_axis: list = [1.,0.,0.]) -> np.ndarray:
    """ 
    Calculate a rotation matrix that aligns a default axis with a given principal axis.
    
    Parameters
    ----------
    p_axis : 1x3 : obj : `np.ndarray`
        principal axis of a gripper model
    norm : float
        l2 norm of the axis vector
    p_axis : 1x3 : obj : `list`
        default axis to align with the principal axis

    Returns
    -------
    R : 3x3 :obj:`numpy.ndarray`
        rotation matrix
    """
    R = np.eye(3, dtype=np.float64)

    unit_vec1 = np.array(d_axis)
    unit_vec2 = p_axis / norm
    v = np.cross(unit_vec1, unit_vec2)
    c = np.dot(unit_vec1, unit_vec2)
    Vmat = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    R = R + Vmat + Vmat @ Vmat / (1 + c + 1e-8)

    return R

def unpacking(sub: list, anchors: list, q_sample: list, centers: list, vectors: list, lock) -> None:
    """ 
    Unpack a list object into multiple sub-lists.
    
    Parameters
    ----------
    sub : 1xN : obj : `list`
        list that will be unpacked
    anchors : 1xN : obj : `list`
        indicies of contact point pairs
    q_sample : 1xN : obj : `list`
        indicies of points within gripper stroke models
    centers : 1xN : obj : `list`
        center points of contact point pairs
    vectors : 1xN : obj : `list`
        direction vectors of contact point pairs
   
    Returns
    -------
    None
    """
    for k in range(len(sub) // 4):
        lock.acquire()
        anchors.append(sub[4*k])
        q_sample.append(sub[4*k + 1])
        centers.append(sub[4*k + 2])
        vectors.append(sub[4*k + 3])
        lock.release()

def get_sides(f_vector: np.ndarray, sides: int, full: bool = True) -> list:
    """
    Approximate a friction cone of a contact.
    
    Parameters
    ----------
    f_vector : Nx3 : obj : `numpy.ndarray`
        force vector along the contact normals
    sides : int
        number of sides for approximating the cone
    full : bool
        whether calculate full side vectors
    
    Returns
    -------
    return_vectors : Mx6 : obj : `list`
        list of all sides of the friction cone
    """
    return_vectors = []
    # Get arbitrary vector to get cross product that should be orthogonal to both
    vector_to_cross = f_vector + np.asarray([1., 2., 3.])
    orth = np.cross(f_vector, vector_to_cross)
    orthg_vector = params['friction_coef'] * orth / np.linalg.norm(orth)
    rot_angle = 2 * np.pi / sides

    for side_num in range(sides):
        RM = R.from_rotvec(rot_angle * side_num * f_vector).as_matrix()
        new_vect = RM @ orthg_vector
        if full:
            new_vect = new_vect + f_vector
        norm_vect = new_vect / np.linalg.norm(new_vect)
        return_vectors.append(norm_vect)

    return return_vectors

def check_maxpts(ML_sample: list) -> int:
    """ 
    Calculate maximum number of points from the list of sampled surfaces.
    
    Parameters
    ----------
    ML_sample : 1xN : obj : `list`
        list of sampled surfaces, grasp maps, and quality metrics 
        
    Returns
    -------
    int : maximum number of points 
    """
    for data in ML_sample:
        max_pts = max(data.shape[0], max_pts)

    return max_pts

class Visualization():
    def __init__(self, pcd) -> None:
        self.pcd = pcd

    def single_cpp(self, contact_p: np.ndarray, surface = None, use_gripper: bool = True) -> list:
        """ 
        Visualize a contact point pair and sampled surfaces.
        
        Parameters
        ----------
        pcd : obj : `open3d.geometry.PointCloud`
            point cloud generated from a mesh file
        contact_p : Nx3 : obj : `numpy.ndarray`
            contact point pair of gripper fingers
        surface : obj : `open3d.geometry.PointCloud`
            sampled surfaces around the contact point pair
        use_gripper: bool
            whether use a simple gripper model for visualization
            
        Returns
        -------
        list2vis : 1XM : obj : `list` 
            list of objects that will be visualized
        """
        list2vis = []

        if surface is not None:
            np.asarray(self.surfaces.colors)[:,:] = [0, 1, 0]   
            list2vis.append(surface)

        if use_gripper:
            gripper = o3d.io.read_triangle_mesh('./objects/gripper.stl')
            gripper.paint_uniform_color([1, 0, 0])
            p_axis = contact_p[0,:] - contact_p[1,:]
            R = rot_matrix(p_axis=p_axis, norm=np.linalg.norm(p_axis))
            gripper.rotate(R, center=(0, 0, 0))
            R = gripper.get_rotation_matrix_from_xyz((0, np.pi*random.uniform(-1, 1)/3, 0))
            gripper.rotate(R, center=(0, 0, 0))

            center = (contact_p[0,:] + contact_p[1,:]) / 2
            gripper.translate((center[0], center[1], center[2]))
            list2vis.append(gripper)
        else:
            mesh_1 = o3d.geometry.TriangleMesh.create_sphere(radius=5, resolution=5).paint_uniform_color([1, 0.7, 0])
            mesh_2 = copy.deepcopy(mesh_1)
            mesh_1.translate((contact_p[0,0], contact_p[0,1], contact_p[0,2]), relative=False)
            mesh_2.translate((contact_p[1,0], contact_p[1,1], contact_p[1,2]), relative=False)
            list2vis.extend([mesh_1, mesh_2])

        return list2vis
    
    def all_centers(self, path_dict: str) -> list:
        """ 
        Visualize the grasp centers of all grasp configurations.
        
        Parameters
        ----------
        pcd : obj : `open3d.geometry.PointCloud`
            point cloud generated from a mesh file
        path_dict : str
            path to the grasp dictionary file (.txt)
            
        Returns
        -------
        list2vis : 1XN : obj : `list` 
            list of objects that will be visualized
        """
        with open(path_dict) as f:
            cpps = eval(f.read())
        cpps = np.asarray(cpps)
        centroids = (cpps[:,:3] + cpps[:,3:]) / 2
        centers = o3d.geometry.PointCloud()
        centers.points = o3d.utility.Vector3dVector(centroids)

        np.asarray(self.pcd.colors)[:,:] = [1, 1, 1]
        list2vis = [centers]
        
        return list2vis

class OtherCS():
    def __init__(self, centers: np.ndarray, vectors: np.ndarray, method: str = 'v2s') -> None:
        self.centers = centers
        self.vectors = vectors
        self.method = method

    def dist_cluster(self, dist_clusters: float, rot_clusters: float) -> list:
        """ 
        Cluster contact point pairs based on their centroids.
        
        Parameters
        ----------
        centers : Nx3 : obj : `numpy.ndarray`
            center points of contact point pairs 
        vectors : Nx3 : obj : `numpy.ndarray`
            direction vectors of contact point pairs

        Returns
        -------
        cluster : 1XN : obj : `list`
            final list of selected indicies
        """
        if self.method == 'v2s':
            g_center = o3d.geometry.PointCloud()
            g_center.points = o3d.utility.Vector3dVector(self.centers)
            min_b = g_center.get_min_bound()
            max_b = g_center.get_max_bound()
            _, _, d_groups = g_center.voxel_down_sample_and_trace(voxel_size=dist_clusters, 
                                                                  min_bound=min_b, max_bound=max_b)
        elif self.method == 'kmean' or self.method == 'kmedoid':
            integer =int(dist_clusters)
            if self.method == 'kmean':
                algo = KMeans(n_clusters=integer, random_state=0).fit(self.centers)
            else:
                distmatrix = euclidean_distances(self.centers)
                algo = KMedoids(n_clusters=integer, method='fasterpam').fit(distmatrix)
            d_groups = [[] for _ in range(integer)]
            for idx, group in enumerate(algo.labels_):
                d_groups[group].append(idx)
        else:
            raise ValueError('Please provide valid method: v2c, kmean, kmedoid')

        cluster = []
        for d_group in tqdm(d_groups):
            temp = self.rot_cluster(d_group, self.vectors[d_group,:], rot_clusters)
            cluster.append(temp)

        return cluster

    def rot_cluster(self, d_group: list, sub_vectors: np.ndarray, rot_clusters: float) -> list:
        """ 
        Cluster contact point pairs based on their direction vectors.
        
        Parameters
        ----------
        d_group : 1xN : obj : `list`
            indicies of grouped samples
        sub_vectors : NX3 : obj : `numpy.ndarray`
            grouped direction vectors of contact point pairs
            
        Returns
        -------
        None
        """
        temp = []

        theta = np.arctan(sub_vectors[:,1] / sub_vectors[:,0])
        L2 = np.linalg.norm(x=sub_vectors[:,:2], ord=2, axis=1)
        psi = np.arctan(L2 / sub_vectors[:,2])
        dummy = np.zeros(len(psi))
        s_coord = np.stack([theta, psi, dummy], axis=1)

        if self.method == 'v2s':
            g_vector = o3d.geometry.PointCloud()
            g_vector.points = o3d.utility.Vector3dVector(s_coord)
            min_b = g_vector.get_min_bound()
            max_b = g_vector.get_max_bound()
            _, _, r_groups = g_vector.voxel_down_sample_and_trace(voxel_size=rot_clusters, 
                                                                  min_bound=min_b, max_bound=max_b)
        else:
            integer = int(rot_clusters)
            if self.method == 'kmean':
                algo = KMeans(n_clusters=integer, random_state=0).fit(s_coord)
            else:
                distmatrix = euclidean_distances(s_coord)
                algo = KMedoids(n_clusters=integer, method='fasterpam').fit(distmatrix)
            r_groups = [[] for _ in range(integer)]
            for idx, group in enumerate(algo.labels_):
                r_groups[group].append(idx)

        for sub in r_groups:
            res = self.find_medoid(s_coord, sub) if len(sub) > 1 else sub[0]
            temp.append(d_group[res])

        return temp

    def find_medoid(self, s_coord: np.ndarray, sub: list) -> int:
        mean = np.mean(s_coord[sub, :], axis=0)
        stack = np.vstack((mean, s_coord[sub, :]))
        find_medoid = o3d.geometry.PointCloud()
        find_medoid.points = o3d.utility.Vector3dVector(stack)

        medoid_tree = o3d.geometry.KDTreeFlann(find_medoid)
        [_, idx, _] = medoid_tree.search_knn_vector_3d(find_medoid.points[0], 2)

        return sub[idx[1] - 1]
