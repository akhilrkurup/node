import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from crisp_py.robot import Robot
from crisp_py.utils.geometry import Pose
from crisp_py.gripper import Gripper, GripperConfig
import socket
import json
import time


# ==========================================
# MONKEY PATCH FOR CRISP_PY BUG
# ==========================================
# This intercepts the ROS message creation to cast np.float64
# to native Python floats, satisfying rclpy's strict type checking
# while keeping the numpy arrays intact for crisp_py's internal math.
_orig_to_ros_msg = Pose.to_ros_msg


def _patched_to_ros_msg(self, *args, **kwargs):
    orig_pos = self.position
    # Temporarily convert to native floats to prevent the AssertionError
    if hasattr(self.position, "__iter__"):
        self.position = [float(x) for x in self.position]

    try:
        msg = _orig_to_ros_msg(self, *args, **kwargs)

        # Catch the orientation quaternion float64 bug as well
        if hasattr(self.orientation, "as_quat"):
            q = self.orientation.as_quat()
            msg.pose.orientation.x = float(q[0])
            msg.pose.orientation.y = float(q[1])
            msg.pose.orientation.z = float(q[2])
            msg.pose.orientation.w = float(q[3])
    finally:
        # Always restore the numpy array so robot.move_to() math works
        self.position = orig_pos

    return msg


Pose.to_ros_msg = _patched_to_ros_msg
# ==========================================

robot = Robot()
robot.wait_until_ready()
traj = np.load("IROS/wiping/processed/node_rollouts_51.npy")[0]
robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

current_pose = robot.end_effector_pose.copy()
rate = robot.node.create_rate(5)
start_pose = robot.end_effector_pose.copy()
traj[:, 2] = start_pose.position[2]
# Pass pure numpy arrays here (removed .tolist())
first_position = traj[0]
first_orientation = start_pose.orientation
first_pose = Pose(position=first_position, orientation=first_orientation)
robot.move_to(position=None, pose=first_pose)
t = 0
for i in range(len(traj)):

    print(i)
    waypoint = traj[i]
    desired_pose = Pose(
        position=waypoint,  # removed .tolist()
        orientation=first_orientation,
    )
    robot.set_target(pose=desired_pose)
    rate.sleep()
    t += 1.0 / 10.0

# robot.home()
robot.shutdown()
