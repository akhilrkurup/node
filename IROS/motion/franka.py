import numpy as np
import time
from scipy.spatial.transform import Rotation
from franky import (
    Robot,
    Affine,
    CartesianState,
    CartesianWaypoint,
    CartesianWaypointMotion,
    CartesianMotion,
)


def execute_trajectory_on_franka(npy_path, robot_ip):
    # 1. Initialize Hardware & Clear Errors
    print(f"Connecting to Franka at {robot_ip}...")
    robot = Robot(robot_ip)
    robot.recover_from_errors()

    # Set conservative dynamics
    robot.relative_dynamics_factor = 0.01

    initial_quat = robot.current_cartesian_state.pose.end_effector_pose.quaternion
    initial_z = robot.current_cartesian_state.pose.end_effector_pose.translation[2]
    print("Captured initial end-effector orientation.")

    # 3. Load the trajectory (N, 3)
    traj = np.load(npy_path)
    if len(traj.shape) != 2 or traj.shape[1] != 3:
        raise ValueError(f"Expected trajectory of shape (N, 3), got {traj.shape}")

    n_points = traj.shape[0]
    print(f"Loaded trajectory with {n_points} waypoints.")

    # 4. Modify Trajectory Geometry
    # Rewrite the Z component to be exactly 0.1 meters for all points
    traj[:, 2] = initial_z

    # Apply safety offset (Currently [0,0,0] but kept for future use)
    workspace_offset = np.array([0, 0.0, 0.0])
    safe_traj = traj + workspace_offset

    # 5. Move to the Start Point
    start_pos = safe_traj[0].tolist()
    start_pose = Affine(start_pos, initial_quat)

    print(f"\nMoving to start position: {start_pos}")
    start_motion = CartesianMotion(start_pose)
    robot.move(start_motion)

    # 6. Wait for human confirmation
    input(
        "\n✅ Reached start point. Press [ENTER] to execute the rest of the trajectory..."
    )

    # 7. Build the Waypoint List (Without explicit velocities)
    waypoints = []
    # Start loop from index 1 since we are already at index 0
    for i in range(1, n_points):
        pos = safe_traj[i].tolist()

        # State contains only position and orientation, no twist/velocity
        state = CartesianState(pose=Affine(pos, initial_quat))
        waypoints.append(CartesianWaypoint(state))


    # 8. Execute the remaining trajectory
    print("Executing waypoint motion... Keep your hand on the emergency stop.")
    motion = CartesianWaypointMotion(waypoints)
    robot.move(motion)

    print("Trajectory completed successfully.")


if __name__ == "__main__":
    # Replace with your actual paths and IP
    NPY_FILE = "/home/akhil-hiro/Documents/GitHub/node/wiping_data/bc_rollout_bad.npy"
    FRANKA_IP = "192.168.1.11"

    execute_trajectory_on_franka(NPY_FILE, FRANKA_IP)
