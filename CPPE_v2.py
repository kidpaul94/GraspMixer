import os
import time
import math
import copy
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from multiprocessing import Process, Lock, Manager

from quality import GraspMetrics
from gripper_config import params
from CPPE_utils import rot_matrix, unpacking, OtherCS

class Gripper():
    def __init__(self, p1: np.ndarray, p2: np.ndarray, p2_p1: np.ndarray, dist: np.float64, pcd = None) -> None:
        self.p1, self.p2 = p1, p2
        self.p2_p1, self.dist = p2_p1, dist
        self.pcd = pcd

    def stroke_model(self):
        """ 
        Generate an intial gripper stroke and collision models that do not
        align with a contact point pair.

        Parameters
        ----------
        finger_dims : 1xN : obj : `list`
            radius (mm & m), thickness (mm), and length of the fingers to use
        dist : float
            set of torques on object in object basis

        Returns
        -------
        gripper : obj : `open3d.geometry.OrientedBoundingBox`
            initial gripper stroke model
        collision : obj : `open3d.geometry.OrientedBoundingBox`
            initial gripper collision model
        """
        radius = params['finger_dims'][0]
        width = self.dist + params['tolerance']
        init = np.array([[0.,0.,0.],[0.,0.,radius],[width,0.,0.],[width,0.,radius],
                            [0.,radius,0.],[0.,radius,radius],[width,radius,0.],[width,radius,radius]])
        init_box = o3d.utility.Vector3dVector(init)
        gripper = o3d.geometry.OrientedBoundingBox.create_from_points(points=init_box)

        width = params['max_width'] + params['finger_dims'][1] + params['tolerance']
        adjust = np.array([[0.,0.,0.],[0.,0.,radius],[width,0.,0.],[width,0.,radius],
                        [0.,radius,0.],[0.,radius,radius],[width,radius,0.],[width,radius,radius]])
        adjust_box = o3d.utility.Vector3dVector(adjust)
        collision = o3d.geometry.OrientedBoundingBox.create_from_points(points=adjust_box)

        return gripper, collision

    def gripper_localizer(self):
        """ 
        Align a gripper stroke and collision models with a contact point pair.

        Parameters
        ----------
        p1 : 1x3 : obj : `numpy.ndarray`
            1st contact point
        p2 : 1x3 : obj : `numpy.ndarray`
            2nd contact point
        dist : 1x3 : obj : `numpy.ndarray`
            distance between two contact points

        Returns
        -------
        gripper : obj : `open3d.geometry.OrientedBoundingBox`
            aligned gripper stroke model
        collision : obj : `open3d.geometry.OrientedBoundingBox`
            aligned gripper stroke model
        center : 1x3 : obj : `numpy.ndarray`
            center points of contact point pairs
        """
        gripper, collision = self.stroke_model()
        R = rot_matrix(self.p2_p1, self.dist)
        center = (self.p1 + self.p2) / 2
        gripper.rotate(R=R, center=center)
        collision.rotate(R=R, center=center)
        gripper.translate(translation=center, relative=False)
        collision.translate(translation=center, relative=False)
        
        return gripper, collision, center

    def check_stroke(self):
        """ 
        Check whether there is any collision around gripper fingers.

        Parameters
        ----------
        pcd : obj : `open3d.geometry.PointCloud`
            point cloud generated from a mesh file

        Returns
        -------
        gripper : bool
            grasp map
        org : 1xM : obj : `list`
            indicies of points within gripper stroke models
        center : 1x3 : obj : `numpy.ndarray`
            center points of contact point pairs
        """
        gripper, collision, center = self.gripper_localizer()
        org = gripper.get_point_indices_within_bounding_box(self.pcd.points)
        ext = collision.get_point_indices_within_bounding_box(self.pcd.points)

        return len(ext) == len(org), org, center
    
    @staticmethod
    def finger_model(gripper, collision, approach: np.ndarray, switch: bool):
        """ 
        Generate collision models of gripper fingers.

        Parameters
        ----------
        gripper : obj : `open3d.geometry.OrientedBoundingBox`
            initial gripper stroke model
        collision : obj : `open3d.geometry.OrientedBoundingBox`
            initial gripper collision model
        finger_dims : 1xN : obj : `list`
            radius (mm & m), thickness (mm), and length of the fingers to use
        approach : 3x1 : obj : `numpy.ndarray`
            approach vector of a gripper
        switch : bool
            switch to adjust bounding box indicies

        Returns
        -------
        finger_1 : obj : `open3d.geometry.OrientedBoundingBox`
            finger_1 collision model
        finger_2 : obj : `open3d.geometry.OrientedBoundingBox`
            finger_2 collision model
        collision_t : obj : `open3d.geometry.OrientedBoundingBox`
            gripper body collision model
        """
        gripper_t = copy.deepcopy(gripper)
        collision_t = copy.deepcopy(collision)
        trans = params['finger_dims'][2] * approach
        gripper_t.translate(translation=trans, relative=True)
        collision_t.translate(translation=trans, relative=True)

        pts_1 = np.asarray(gripper.get_box_points())
        pts_2 = np.asarray(collision.get_box_points())
        pts_3 = np.asarray(gripper_t.get_box_points())
        pts_4 = np.asarray(collision_t.get_box_points())

        if switch:
            group_1 = np.asarray([pts_1[3,:], pts_1[5,:], pts_2[3,:], pts_2[5,:], 
                                    pts_3[3,:], pts_3[5,:], pts_4[3,:], pts_4[5,:]])
            group_2 = np.asarray([pts_1[4,:], pts_1[6,:], pts_2[4,:], pts_2[6,:], 
                                    pts_3[4,:], pts_3[6,:], pts_4[4,:], pts_4[6,:]])
        else:
            group_1 = np.asarray([pts_1[1,:], pts_1[7,:], pts_2[0,:], pts_2[2,:], 
                                    pts_3[1,:], pts_3[7,:], pts_4[0,:], pts_4[2,:]])
            group_2 = np.asarray([pts_1[4,:], pts_1[6,:], pts_2[1,:], pts_2[7,:], 
                                    pts_3[4,:], pts_3[6,:], pts_4[1,:], pts_4[7,:]])
        box_1 = o3d.utility.Vector3dVector(group_1)
        finger_1 = o3d.geometry.OrientedBoundingBox.create_from_points(points=box_1)
        box_2 = o3d.utility.Vector3dVector(group_2)
        finger_2 = o3d.geometry.OrientedBoundingBox.create_from_points(points=box_2)

        return finger_1, finger_2, collision_t

    def collision_approx(self, side : int = 12, threshold : int = 20) -> list:
        """ 
        Rotate a proxy gripper model around a direction vector of 2 CPP to
        check potential collisions between the object and the gripper.
        
        Parameters
        ----------
        side : int
            number to discretize a full rotation along a rotation vector
            a rotation vector
        threshold : int
            threshold to determine whether a gripper collide with an object

        Returns
        -------
        res : 1XN : obj : `list`
            Possible approach vectors of a gripper without collisions
        """
        d_vector = self.p2_p1 / self.dist
        gripper, collision, center = self.gripper_localizer()
        RM = R.from_rotvec(2 * np.pi / side * d_vector).as_matrix() 
        res = []
        
        for _ in range(side):
            gripper.rotate(R=RM, center=center)
            collision.rotate(R=RM, center=center)
            pts = np.asarray(gripper.get_box_points())
            vertex = pts[3,:] - pts[0,:]
            vertex_norm = np.linalg.norm(vertex)
            switch = math.isclose(vertex_norm, params['finger_dims'][0], abs_tol=1e-2)
            print(vertex_norm)
            vector = vertex if switch else pts[1,:] - pts[0,:]
            vector = vector / params['finger_dims'][0] 

            finger_1, finger_2, collision_t = self.finger_model(gripper, collision, vector, switch)
            ext_1 = collision_t.get_point_indices_within_bounding_box(self.pcd.points)
            ext_2 = finger_1.get_point_indices_within_bounding_box(self.pcd.points)
            ext_3 = finger_2.get_point_indices_within_bounding_box(self.pcd.points)

            if len(ext_1) + len(ext_2) + len(ext_3) < threshold:        
                res.append(vector.tolist())

        return res

class CPPE():
    def __init__(self, path: str = './objects/pcds/4096_obj_05.pcd') -> None:
        self.pcd = o3d.io.read_point_cloud(path)
        self.tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.pts = np.asarray(self.pcd.points)
        self.normals = np.asarray(self.pcd.normals)

    def sample_pair(self, i: int, sample, lock) -> None:
        """ 
        Sameple contact point pairs based on antipodal grasp criterion.
        
        Parameters
        ----------
        pcd : obj : `open3d.geometry.PointCloud`
            point cloud generated from a mesh file
        tree : obj : `open3d.geometry.KDTreeFlann`
            KDTree with FLANN for nearest neighbor search
        pts : Nx3 : obj : `numpy.ndarray`
            points of a generated point cloud 
        normals : Nx3 : obj : `numpy.ndarray`
            normals of a generated point cloud
    
        Returns
        -------
        None
        """
        [_, idx, _] = self.tree.search_radius_vector_3d(self.pcd.points[4*i], params['max_width'])
        temp = []

        for j in range(1, len(idx)):
            p2_p1 = self.pts[idx[j],:] - self.pts[idx[0],:]
            dist = np.linalg.norm(p2_p1, ord=2, axis=0)
            dot_n1d = -np.dot(self.normals[idx[0],:], p2_p1) / dist
            dot_n2d = np.dot(self.normals[idx[j],:], p2_p1) / dist            

            """
            This is a parameter to tune: 0.9396926 = cos(20), 0.9659258 = cos(15)
            mu: tan(20) = 0.364, tan(15) = 0.268
            """
            if dot_n1d > 0.9659258 and dot_n2d > 0.9659258:
                gripper = Gripper(self.pts[idx[0],:], self.pts[idx[j],:], p2_p1, dist, self.pcd)
                isvalid, q_sample, center = gripper.check_stroke()
                if isvalid:
                    temp.extend([[idx[0], idx[j]], q_sample, center, p2_p1 / dist])

        if len(temp) > 1:
            lock.acquire()
            sample.append(temp)
            lock.release()

    def contact_pp_estimation(self, num_pts: int, lock):
        """ 
        Use multiprocessing to speed up the sampling process.

        Parameters
        ----------
        None
        
        Returns
        -------
        pts : Nx3 : obj : `numpy.ndarray`
            points of a generated point cloud 
        normals : Nx3 : obj : `numpy.ndarray`
            normals of a generated point cloud
        pcd : obj : `open3d.geometry.PointCloud`
            point cloud generated from a mesh file
        """
        with Manager() as manager:
            sample = manager.list()
            processes = []
            for i in range(num_pts // 4):
                args = (i, sample, lock)
                p = Process(target=self.sample_pair, args=args)
                p.start()
                processes.append(p)

            for process in processes:
                process.join()

            sample = list(sample)

        return sample, self.pts, self.normals, self.pcd

class Voxel2stgCM():
    def __init__(self, centers: np.ndarray, vectors: np.ndarray) -> None:
        self.centers = centers
        self.vectors = vectors

    def dist_cluster(self, lock, dist_clusters: float = 10., 
                     rot_clusters: float = 0.1745) -> list:
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
        g_center = o3d.geometry.PointCloud()
        g_center.points = o3d.utility.Vector3dVector(self.centers)
        min_b = g_center.get_min_bound()
        max_b = g_center.get_max_bound()
        _, _, d_groups = g_center.voxel_down_sample_and_trace(voxel_size=dist_clusters, 
                                                              min_bound=min_b, max_bound=max_b)

        with Manager() as manager:
            cluster = manager.list()
            processes = []

            for d_group in d_groups:
                args = (lock, d_group, self.vectors[d_group,:], rot_clusters, cluster)
                p = Process(target=self.rot_cluster, args=args)
                p.start()
                processes.append(p)

            for process in processes:
                process.join()

            cluster = list(cluster)

        return cluster

    def rot_cluster(self, lock, d_group: list, sub_vectors: np.ndarray, 
                    rot_clusters: float, cluster) -> None:
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

        g_vector = o3d.geometry.PointCloud()
        g_vector.points = o3d.utility.Vector3dVector(s_coord)
        min_b = g_vector.get_min_bound()
        max_b = g_vector.get_max_bound()
        _, _, r_groups = g_vector.voxel_down_sample_and_trace(voxel_size=rot_clusters, min_bound=min_b, 
                                                              max_bound=max_b)

        for sub in r_groups:
            res = self.find_medoid(s_coord, sub) if len(sub) > 1 else sub[0]
            temp.append(d_group[res])

        lock.acquire()
        cluster.append(temp)
        lock.release()

    def find_medoid(self, s_coord: np.ndarray, sub: list) -> int:
        mean = np.mean(s_coord[sub, :], axis=0)
        stack = np.vstack((mean, s_coord[sub, :]))
        find_medoid = o3d.geometry.PointCloud()
        find_medoid.points = o3d.utility.Vector3dVector(stack)

        medoid_tree = o3d.geometry.KDTreeFlann(find_medoid)
        [_, idx, _] = medoid_tree.search_knn_vector_3d(find_medoid.points[0], 2)

        return sub[idx[1] - 1]

class Preprocess():
    def __init__(self, contact_p: np.ndarray, pts_sample: np.ndarray, normal_sample: np.ndarray) -> None:
        self.contact_p = contact_p 
        self.surface_pts, self.surface_normals = self.Q_preprocess(pts_sample, normal_sample)
        self.max_d = np.sqrt(0.25 * params['max_width']**2 + 2 * params['finger_dims'][0]**2)

    def Q_preprocess(self, pts_sample: np.ndarray, normal_sample: np.ndarray) -> np.ndarray:
        """ 
        Refine received data to obtain surface points and normals.
        
        Parameters
        ----------
        contact_p : NX3 : obj : `numpy.ndarray`
            contact point pair of gripper fingers
        pts_sample : NX3 : obj : `numpy.ndarray`
            points in a gripper stroke model
        normal_sample : NX3 : obj : `numpy.ndarray`
            normal vectors associated with points in a gripper stroke model

        Returns
        -------
        surface_pts : 1XN : obj : `list`
            surface points (numpy.ndarray) around the contact points
        surface_normals : 1XN : obj : `list`
            surface normals (numpy.ndarray) around the contact points
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_sample)
        pcd.normals = o3d.utility.Vector3dVector(normal_sample)

        camera = 0.2 * (self.contact_p[0,:] - self.contact_p[1,:])
        radius = 100 * np.linalg.norm(camera)

        _, pt_map_1 = pcd.hidden_point_removal(self.contact_p[0,:] + camera, radius)
        _, pt_map_2 = pcd.hidden_point_removal(self.contact_p[1,:] - camera, radius)
        temp_1 = pcd.select_by_index(pt_map_1)
        temp_2 = pcd.select_by_index(pt_map_2)

        filtered_1, _ = temp_1.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        filtered_2, _ = temp_2.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        surface_pts = [np.asarray(filtered_1.points), np.asarray(filtered_2.points)]
        surface_normals = [np.asarray(filtered_1.normals), np.asarray(filtered_2.normals)] 

        return surface_pts, surface_normals

    def ML_preprocess(self, centroid: np.ndarray) -> np.ndarray:
        """ 
        Preprocess received data to generate input for deep learning algorithm
        
        Parameters
        ----------
        surface_pts : 1XN : obj : `list`
            surface points (numpy.ndarray) around the contact points
        surface_normals : 1XN : obj : `list`
            surface normals (numpy.ndarray) around the contact points
        centroid : Nx3 : obj : `numpy.ndarray`
            centroid of a contact point pair

        Returns
        -------
        Nx6 : obj : `numpy.ndarray` : input data of deep learning algorithm
        """
        
        normalized_1 = (self.surface_pts[0] - centroid) / self.max_d
        normalized_2 = (self.surface_pts[1] - centroid) / self.max_d
        combined_1 = np.hstack((normalized_1, self.surface_normals[0]))
        combined_2 = np.hstack((normalized_2, self.surface_normals[1]))

        return np.vstack((combined_1, combined_2))

def postprocess(idx: int, anchors: list, q_sample: list, centroids: np.ndarray, CoM: np.ndarray,
                pts: np.ndarray, normals: np.ndarray) -> list:
    """ 
    Postprocess received data to generate input for deep learning algorithm
    
    Parameters
    ----------
    idx : int
        index to select values fro lists
    anchors : 1xN : obj : `list`
        indicies of contact point pairs
    q_sample : 1xN : obj : `list`
        indicies of points within gripper stroke models
    centroids : Nx3 : obj : `numpy.ndarray`
        centroids of contact point pairs
    CoM : 1x3 : obj : `numpy.ndarray`
        object's center of mass
    pts : Nx3 : obj : `numpy.ndarray`
        points of a generated point cloud 
    normals : Nx3 : obj : `numpy.ndarray`
        normals of a generated point cloud
    grasp_dict : 1xN : obj : `list`
        xyz values of contact points and associated grasp metrics
    ML_sample : 1xN : obj : `list`
        input data of deep learning algorithm

    Returns
    -------
    res_1 : Nx6 : obj : `numpy.ndarray`
        input data of deep learning algorithm
    res_2 : 1xN : obj : `list`
        xyz values of contact points and associated grasp metrics
    res_3 : Mx3 : obj : `numpy.ndarray`
        list of all information related to the current grasp
    """
    contact_p = pts[anchors[idx]]
    contact_n = normals[anchors[idx]]
    centroid = centroids[idx]

    prep = Preprocess(contact_p, pts[q_sample[idx],:], normals[q_sample[idx],:])        
    metric = GraspMetrics(centroid, contact_p, contact_n, pts, CoM)        
    finger_joint = np.linalg.norm(contact_p[0,:] - centroid)
 
    res_1 = prep.ML_preprocess(centroid)
    res_2 = [contact_p[0,0], contact_p[0,1], contact_p[0,2], 
             contact_p[1,0], contact_p[1,1], contact_p[1,2]]
    res_3 = metric.Q_combined([finger_joint, finger_joint], prep.surface_normals)

    return res_1, res_2, res_3

def parse_args(argv = None) -> None:
    parser = argparse.ArgumentParser(description='CPPE')
    parser.add_argument('--object_path', default='./objects/pcds/4096_obj_05.pcd', type=str,
                        help='path of an object we wish to evaluate.')
    parser.add_argument('--num_pts', default=4096, type=int,
                        help='number of points in the .pcd file.')
    parser.add_argument('--CoM', default=[0., 0., 0.], type=float, nargs='+',
                        help='object center of mass.')
    parser.add_argument('--clustering', default='v2s', type=str,
                        help='clustering algorithm: v2s, v2sm, kmean, kmedoid.')
    parser.add_argument('--k_dist', default=20., type=float,
                        help='value that controls the number of dist_clusters.')
    parser.add_argument('--k_rot', default=0.3491, type=float,
                        help='value that controls the number of rot_clusters.')
    parser.add_argument('--side', default=12, type=int,
                        help='value to discretize a full rotation along a rotation vector.')
    parser.add_argument('--threshold', default=20, type=int,
                        help='value to determine whether a gripper collide with an object.')

    global pargs
    pargs = parser.parse_args(argv)

if __name__ == "__main__":
    parse_args()
    lock = Lock()
    filepath = pargs.object_path
    CoM = np.array(pargs.CoM)
    start = time.time()

    print(f'Start sampling grasps...')
    sample, pts, normals, pcd = CPPE(filepath).contact_pp_estimation(pargs.num_pts, lock)
    print(f'Done. Time elapsed: {time.time()-start} sec')
    print(f'Start unpacking grasps samples...')
    start = time.time()

    with Manager() as manager:
        anchors, q_sample = manager.list(), manager.list()
        centers, vectors = manager.list(), manager.list()    
        processes = []
        for sub in sample:
            args = (sub, anchors, q_sample, centers, vectors, lock)
            p = Process(target=unpacking, args=args)
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        anchors, q_sample = list(anchors), list(q_sample)
        centers, vectors = list(centers), list(vectors)

    print(f'Done. Time elapsed: {time.time()-start} sec')
    print(f'Initial # of samples: {len(centers)}')
    print(f'Start clustering grasps samples...')
    start = time.time()

    centers = np.asarray(centers)
    vectors = np.asarray(vectors)
    if pargs.clustering == 'v2sm':
        algo = Voxel2stgCM(centers=centers, vectors=vectors)
        cluster = algo.dist_cluster(lock, dist_clusters=pargs.k_dist, rot_clusters=pargs.k_rot)
    else:
        algo = OtherCS(centers=centers, vectors=vectors, method=pargs.clustering)
        cluster = algo.dist_cluster(dist_clusters=pargs.k_dist, rot_clusters=pargs.k_rot)
    flatten = []

    for cpp_idx in cluster:
        flatten.extend(cpp_idx)

    print(f'Done. Time elapsed: {time.time()-start} sec')
    print(f'Start calculating grasp metrics...')
    start = time.time()

    ML_sample, grasp_dict, Qs = [], [], []
    for idx in flatten:
        res_1, res_2, res_3 = postprocess(idx, anchors, q_sample, 
                                          centers, CoM, pts, normals)
        ML_sample.append(res_1)
        grasp_dict.append(res_2)
        Qs.append(res_3)
    
    approach_vectors = []
    for cpp in grasp_dict:
        cpp = np.asarray(cpp)
        p2_p1 = cpp[3:] - cpp[:3]
        dist = np.linalg.norm(p2_p1)
        gripper = Gripper(p1=cpp[:3], p2=cpp[3:], p2_p1=p2_p1, dist=dist, pcd=pcd)
        res = gripper.collision_approx(side=pargs.side, threshold=pargs.threshold)
        approach_vectors.append(res)

    print(f'Done. Time elapsed: {time.time()-start} sec')
    print(f'Total # of CPPs: {len(grasp_dict)}')

    file = os.path.basename(filepath)
    filename, _ = os.path.splitext(file)

    directory = f'./objects/dicts/{filename}'
    if not os.path.exists(directory):
        print(f'Generate {filename} folder...')
        os.mkdir(directory)

    with open(f'{directory}/{filename}_cpps.txt', 'w') as output:
        print(f'Generate {filename} grasp dictionaries...')
        output.write(repr(grasp_dict))
        output.close()

    with open(f'{directory}/{filename}_aprvs.txt', 'w') as output:
        print(f'Generate {filename} approach_vectors dictionaries...')
        output.write(repr(approach_vectors))
        output.close()

    print(f'Save associated ML data as .npy format...')
    for num in range(len(ML_sample)):
        np.save(f'{directory}/pts_{num:04d}.npy', ML_sample[num])

    print(f'Save associated misc data as .npy format...')
    for num in range(len(Qs)):
        np.save(f'{directory}/misc_{num:04d}.npy', Qs[num])
