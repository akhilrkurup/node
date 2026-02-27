import numpy as np
import math
from scipy.spatial.transform import Rotation
from franky import (
    Robot,
    Affine,
    Twist,
    CartesianState,
    CartesianWaypoint,
    CartesianWaypointMotion,
    RelativeDynamicsFactor,
)


def execute_trajectory_on_franka(npy_path, robot_ip, dt=0.01):
    # 1. Load the trajectory (N, 3)
    traj = np.load(npy_path)
    if len(traj.shape) != 2 or traj.shape[1] != 3:
        raise ValueError(f"Expected trajectory of shape (N, 3), got {traj.shape}")

    n_points = traj.shape[0]
    print(f"Loaded trajectory with {n_points} waypoints.")

    # 2. Safety Offset (CRITICAL)
    # The trajectory from the Neural Network might be centered around the origin (0,0,0).
    # On the Franka, (0,0,0) is physically inside the base of the robot.
    # You MUST shift the trajectory to a safe workspace in front of the robot.
    # Adjust these values based on your physical table height and reach!
    workspace_offset = np.array([0.4, 0.0, 0.2])  # 40cm forward, 20cm above base
    safe_traj = traj + workspace_offset

    # 3. Compute target velocities to prevent stuttering
    # Ruckig needs to know the velocity at each waypoint to maintain continuous flow
    velocities = np.gradient(safe_traj, axis=0) / dt

    # 4. Define End-Effector Orientation
    # We assign a fixed orientation for the rollout (e.g., pointing straight down).
    # Franka's end-effector pointing down is roughly a 180-degree rotation around X or Y.
    quat = Rotation.from_euler("xyz", [math.pi, 0, 0]).as_quat()

    # 5. Build the continuous waypoint list
    waypoints = []
    for i in range(n_points):
        # Extract position and linear velocity for this step
        pos = safe_traj[i].tolist()
        lin_vel = velocities[i].tolist()

        # Zero angular velocity since we want to maintain the fixed downward orientation
        ang_vel = [0.0, 0.0, 0.0]

        # For the very last waypoint, we must force the velocity to exactly 0
        # so the robot comes to a complete, safe stop at the end of the motion.
        if i == n_points - 1:
            lin_vel = [0.0, 0.0, 0.0]

        # Construct the state and waypoint
        state = CartesianState(pose=Affine(pos, quat), velocity=Twist(lin_vel, ang_vel))
        waypoints.append(CartesianWaypoint(state))

    # 6. Initialize Hardware & Execute
    print(f"Connecting to Franka at {robot_ip}...")
    robot = Robot(robot_ip)

    # Clear any prior errors
    robot.recover_from_errors()

    # Set conservative dynamics for the first real-world test
    # This limits the robot to 10% of its maximum velocity, acceleration, and jerk
    robot.relative_dynamics_factor = 0.1

    print("Executing motion... Keep your hand on the emergency stop.")
    motion = CartesianWaypointMotion(waypoints)
    robot.move(motion)

    print("Trajectory completed successfully.")


if __name__ == "__main__":
    # Replace with your actual paths and IP
    NPY_FILE = "generated_rollout.npy"
    FRANKA_IP = "10.90.90.1"

    # dt should match the time delta used during your NODE/BC simulation rollout
    execute_trajectory_on_franka(NPY_FILE, FRANKA_IP, dt=0.005)
