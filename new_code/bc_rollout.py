import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==========================================
# 1. Define Architecture (Must match training)
# ==========================================
class BCPolicy(nn.Module):
    def __init__(self, data_size, width_size, depth):
        super(BCPolicy, self).__init__()
        layers = []
        in_features = data_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width_size))
            layers.append(nn.Tanh())
            in_features = width_size
        layers.append(nn.Linear(in_features, data_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ==========================================
# 2. Simulation Function
# ==========================================
def simulate_trajectory(model, start_point, n_steps=500, dt=0.01):
    """
    Rolls out a trajectory using Euler integration: x_{t+1} = x_t + v*dt
    """
    model.eval()
    current_state = torch.tensor(start_point, dtype=torch.float32)
    trajectory = [current_state.numpy()]

    with torch.no_grad():
        for _ in range(n_steps):
            # 1. Predict velocity from current state
            velocity = model(current_state)

            # 2. Update state (Euler Step)
            current_state = current_state + velocity * dt

            trajectory.append(current_state.numpy())

    return np.array(trajectory)


def simulate_one_loop(
    model, start_point, dt=0.01, leave_dist=0.1, return_dist=0.01, max_steps=5000
):
    """
    Rolls out a trajectory using Euler integration and terminates automatically
    when the state closes the geometric loop.
    """
    model.eval()

    # Initialize tensors
    current_state = torch.tensor(start_point, dtype=torch.float32)
    x0 = current_state.clone()  # Keep a strict reference to the origin

    trajectory = [current_state.numpy()]
    has_left_start = False

    with torch.no_grad():
        for step in range(max_steps):
            # 1. Forward pass & Euler update
            velocity = model(current_state)
            current_state = current_state + velocity * dt
            trajectory.append(current_state.numpy())

            # 2. Compute Euclidean distance to the initial condition
            dist_to_start = torch.norm(current_state - x0).item()

            # 3. State Machine for Loop Closure
            if not has_left_start:
                # Check if we have broken out of the starting neighborhood
                if dist_to_start > leave_dist:
                    has_left_start = True
            else:
                # We have left. Now check if we have returned.
                if dist_to_start < return_dist:
                    print(
                        f"Loop successfully closed at step {step}. Terminating rollout."
                    )
                    break

        else:
            # This triggers only if the for-loop reaches max_steps without breaking
            print(
                f"Warning: Rollout reached max_steps ({max_steps}) without closing the loop. Covariate shift likely caused divergence."
            )

    return np.array(trajectory)


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # Parameters (Must match how you trained)
    DATA_DIM = 3  # Set to 2 if training was 2D
    WIDTH = 64
    DEPTH = 3
    MODEL_PATH = r"C:\Users\akhil\OneDrive\Documents\GitHub\CLF-CBF-NODE\wiping_data\bc_polic_iter3300.pth"

    # Initialize and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(DATA_DIM, WIDTH, DEPTH).to(device)

    # Load weights
    # If you saved the full dict, use ['model_state_dict'], otherwise load directly
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.to("cpu")  # Move to CPU for simple inference loop

    # Define a starting point (Normalized range -0.5 to 0.5)
    # Example: Starting at the 'top' of the wiping motion
    start_pos = [0.2, 0, 0.0]

    # Run Simulation
    print("Simulating rollout...")
    generated_traj = simulate_trajectory(model, start_pos, n_steps=140)
    np.save("./wiping_data/bc_rollout.npy", generated_traj)
    # Visualization
    fig = plt.figure(figsize=(10, 7))
    if DATA_DIM == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            generated_traj[:, 0],
            generated_traj[:, 1],
            "r-",
            label="BC Rollout",
        )
        ax.scatter(
            generated_traj[0, 0],
            generated_traj[0, 1],
            c="g",
            s=100,
            label="Start",
        )
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        ax.plot(generated_traj[:, 0], generated_traj[:, 1], "r-", label="BC Rollout")
        ax.scatter(
            generated_traj[0, 0], generated_traj[0, 1], c="g", s=100, label="Start"
        )

    ax.set_title("Behavior Cloning Inference Rollout")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.legend()
    plt.show()
