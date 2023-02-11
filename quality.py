import numpy as np
from scipy.spatial import ConvexHull, distance

from CPPE_utils import get_sides
from gripper_config import params

class GraspMetrics():
    def __init__(self, centroid: np.ndarray, contact_p: np.ndarray, contact_n: np.ndarray, pts: np.ndarray, CoM: np.ndarray) -> None:
        self.pts, self.CoM = pts, CoM
        self.centroid, self.contact_p, self.contact_n = centroid, contact_p, contact_n
        self.forces = -params['gripper_force'] * contact_n
        self.max_d = max(np.linalg.norm(pts - CoM, axis=1))
        self.torques = np.cross(contact_p - CoM, self.forces) / self.max_d
        self.g_wrenches = self.gws_pyramid()

    def gws_pyramid(self, pyramid_sides: int = 8) -> list:
        """
        Generate potential wrenches that can be applied to contact points.
        
        Parameters
        ----------
        contact_p : Nx3 : obj : `numpy.ndarray`
            contact point pair of gripper fingers
        contact_n : Nx3 : obj : `numpy.ndarray`
            surface normals at the contact points
        CoM : 1x3 : obj : `numpy.ndarray`
            center of mass of the object
        max_d : float
            maximum distance from the object's CoM
        pyramid_sides : int
            number of sides of the cone
        
        Returns
        -------
        force_torque : Mx6 : obj : `list`
            list of potential wrenches that can be applied to contact points
        """
        force_torque = []
        for idx, data in enumerate(self.contact_p):
            force_vector = -self.contact_n[idx,:]

            if np.linalg.norm(force_vector) > 0:
                new_vectors = get_sides(force_vector, pyramid_sides)
                radius_to_contact = (data - self.CoM)

                for pyramid_vector in new_vectors:
                    torque_vector = np.cross(radius_to_contact, pyramid_vector) / self.max_d
                    force_torque.append(np.hstack((pyramid_vector, torque_vector)))

        return force_torque

    @staticmethod
    def QA3(G: np.ndarray) -> float:
        """
        Grasp Isotropy Index - ratio of "weakest" wrench that the grasp can exert to the "strongest" one.
        
        Parameters
        ----------
        G : 6xM : obj : `numpy.ndarray`
            grasp map
        
        Returns
        -------
        isotropy : float
            value of grasp isotropy metric
        """
        _, S, _ = np.linalg.svd(G)
        max_sig = S[0]
        min_sig = S[5]
        isotropy = min_sig / max_sig
        if np.isnan(isotropy) or np.isinf(isotropy):
            return 0

        return isotropy

    def QB1(self) -> float:
        """ 
        Distance between the centroid of the contact polygon and the object's center of mass.
        
        Parameters
        ----------
        centroid : 1x3 : obj : `numpy.ndarray`
            centroid of the contact points
        CoM : 3x1 : obj : `numpy.ndarray`
            object's center of mass
        max_dist : 1x3 : obj : `numpy.ndarray`
            maximum distance from the center of mass of the object 
            to any point in the object's contour

        Returns
        -------
        float : value of the distance-based metric
        """
        distance = np.linalg.norm(self.centroid - self.CoM)

        return 1 - distance / self.max_d

    def QC1(self):
        """ 
        Ferrari & Canny's L1 metric. Also known as the epsilon metric.
        
        Parameters
        ----------
        g_wrenches : Mx6 : obj : `list`
            list of potential wrenches that can be applied to contact points
        
        Returns
        -------
        float : value of the metric
        """
        hull = ConvexHull(points=self.g_wrenches, qhull_options='QJ')
        centroid = []
        for dim in range(0, 6):
            centroid.append(np.mean(hull.points[hull.vertices, dim]))
        shortest_distance = 500000000
        for point in self.g_wrenches:
            point_dist = distance.euclidean(centroid, point)
            if point_dist < shortest_distance:
                shortest_distance = point_dist

        return shortest_distance / np.sqrt(2)

    def QC2(self):
        """ 
        Volume of a GWS.
        
        Parameters
        ----------
        g_wrenches : Mx6 : obj : `list`
            list of potential wrenches that can be applied to contact points
        
        Returns
        -------
        float : value of the metric
        """
        convex_hull = ConvexHull(points=self.g_wrenches, qhull_options='QJ')
        return convex_hull.volume

    @staticmethod
    def QD1(finger_joint: list) -> float:
        """ 
        Posture of hand finger joints - how far each joint is from its maximum limits.
        
        Parameters
        ----------
        finger_joint : 1xN : obj : `list`
            current posture of hand finger joints
        joint_lims : 1xN : obj : `list`
            joint limit of hand finger

        Returns
        -------
        float : value of the joint-based metric
        """
        sum = 0
        for posture in finger_joint:
            numerator = posture - params['joint_lims'][0]
            denominator = params['joint_lims'][1] - params['joint_lims'][0]
            sum += numerator / denominator
        
        return 1 - sum / len(finger_joint)

    def QF1(self, surface_normals: list) -> float:
        """ 
        Flatness of areas around the contact points.
        
        Parameters
        ----------
        contact_n : Nx3 : obj : `numpy.ndarray`
            surface normals at the contact points
        surface_normals : 1XN : obj : `list`
            surface normals (numpy.ndarray) around the contact points

        Returns
        -------
        float : value of the flatness-based metric
        """
        res = div = 0
        for idx in range(len(surface_normals)):
            temp = np.dot(surface_normals[idx], self.contact_n[idx,:])
            div += np.shape(surface_normals[idx])[0]
            res += np.sum(temp)
        
        return res / div

    def QF2(self) -> float:
        """ 
        Cosine similarity between surface normals and finger closing directions.
        
        Parameters
        ----------
        contact_p : Nx3 : obj : `numpy.ndarray`
            contact point pair of gripper fingers
        contact_n : Nx3 : obj : `numpy.ndarray`
            surface normals at the contact points

        Returns
        -------
        float : value of the cosine similarity-based metric
        """
        dir = self.contact_p[1,:] - self.contact_p[0,:]
        dist = np.linalg.norm(dir, ord=2, axis=0)
        dot_n1d = -np.dot(self.contact_n[0,:], dir) / dist
        dot_n2d = np.dot(self.contact_n[1,:], dir) / dist  

        return dot_n1d * dot_n2d

    def Q_combined(self, finger_joint: float, surface_normals: list) -> np.ndarray:
        """ 
        Combine all information related to the current grasp and return them as a list.
        
        Parameters
        ----------
        finger_joint : 1xN : obj : `list`
            current posture of hand finger joints
        surface_normals : 1XN : obj : `list`
            surface normals (numpy.ndarray) around the contact points
        forces : Nx3 : obj : `numpy.ndarray`
            set of forces on object in object basis
        torques : Nx3 : obj : `numpy.ndarray`
            set of torques on object in object basis
        contact_p : Nx3 : obj : `numpy.ndarray`
            contact point pair of gripper fingers
        contact_n : Nx3 : obj : `numpy.ndarray`
            surface normals at the contact points
        CoM : 1x3 : obj : `numpy.ndarray`
            center of mass of the object
        friction_coefficient : float
            friction coefficient of the gripper  

        Returns
        -------
        combined : Mx3 : obj : `numpy.ndarray`
            list of all information related to the current grasp
        """  
        # quality metrics + object mass
        Qs = np.array([self.QB1(), self.QC1(), self.QD1(finger_joint),
                       self.QF1(surface_normals), self.QF2(), 1.]).reshape(-1,3)
        radius = params['finger_dims'][0] * 0.001
        # finger radius, object friction_coef, gripper friction_coef
        others = np.array([radius, 0.42, params['friction_coef']])
        combined = np.vstack((Qs, self.forces, self.torques, others))
        
        return combined
