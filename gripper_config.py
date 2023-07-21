"""
Physical parameters associated with a gripper model.

Parameters
----------
torque_scaling : float
    torque_scaling
friction_coef : float
    minimum friction coefficient of a gripper
finger_dims : 1xN : obj : `list`
    radius (mm) and thickness (mm) of the fingers to use
tolerance : float
    dimensional tolerance of a gripper model
gripper_force : float
    maximum force that each gripper finger can apply 
joint_lims : 1xN : obj : `list`
    joint limits of each gripper finger
max_width : float
    maximum width between gripper fingers
body_side : float
    side length (mm) of a gripper body
"""

params = {
    'torque_scaling': 1., 
    'friction_coef': 0.268, 
    'finger_dims': [20., 30., 75.],
    'tolerance': 5.,
    'gripper_force': 100.,
    'joint_lims' : [0., 28.5],
    'max_width' : 57.,
    'body_side' : 70.,
    }
