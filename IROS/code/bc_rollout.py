import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os


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


def sample_from_ball(center, radius, num_samples, dim=3):
    """
    Samples points uniformly from an n-dimensional ball.
    """
    points = []
    for _ in range(num_samples):
        # 1. Sample uniformly from the surface of a unit sphere
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)

        # 2. Sample radius taking the volume expansion into account
        u = np.random.rand()
        r = radius * (u ** (1.0 / dim))

        # 3. Translate to center
        points.append(center + r * direction)
    return points


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    plot_2d = True
    # Parameters (Must match how you trained)
    DATA_DIM = 3  # Set to 2 if training was 2D
    WIDTH = 64
    DEPTH = 3
    master_folder = "./IROS/stirring/raw"
    MODEL_PATH = os.path.join(master_folder, r"bc_policy\bc_policy_iter09999.pth")
    # Initialize and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(DATA_DIM, WIDTH, DEPTH).to(device)

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.to("cpu")  # Move to CPU for simple inference loop

    # Define the exact starting point
    exact_start_pos = np.array([0.625, -0.03, 0.0])

    # Generate 50 additional starting points within a 0.1 radius ball
    perturbation_radius = 0.001
    sampled_starts = sample_from_ball(
        exact_start_pos, perturbation_radius, num_samples=50, dim=DATA_DIM
    )

    # Combine into a single list of 51 starting points
    all_starts = [exact_start_pos] * 51  # + sampled_starts

    # Run Simulation for all 51 points
    print(f"Simulating 51 rollouts...")
    all_trajectories = []
    for pt in all_starts:
        traj = simulate_trajectory(model, pt, n_steps=70, dt=0.01)
        all_trajectories.append(traj)

    # Convert to a single numpy array of shape (51, n_steps+1, DATA_DIM)
    all_trajectories_np = np.array(all_trajectories)

    # Save all trajectories
    save_path = os.path.join(master_folder, "bc_rollouts_51.npy")
    np.save(save_path, all_trajectories_np)
    print(f"Saved 51 rollouts to {save_path} with shape {all_trajectories_np.shape}")

    # ==========================================
    # 4. Visualization
    # ==========================================
    fig = plt.figure(figsize=(10, 7))

    # Select indices to plot: Index 0 (exact start) + 3 random indices from the 50 samples
    plot_indices = [0] + random.sample(range(1, 51), 3)
    colors = ["r", "b", "c", "m"]  # Red for exact, others for perturbed
    labels = ["Exact Start"] + ["Perturbed Start"] * 3

    if DATA_DIM == 3 and not plot_2d:
        ax = fig.add_subplot(111, projection="3d")
        for idx, c, label in zip(plot_indices, colors, labels):
            traj = all_trajectories_np[idx]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, label=label, alpha=0.8)
            ax.scatter(
                traj[0, 0], traj[0, 1], traj[0, 2], color="g", s=50, edgecolors="k"
            )  # Start marker

        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        for idx, c, label in zip(plot_indices, colors, labels):
            traj = all_trajectories_np[idx]
            ax.plot(traj[:, 0], traj[:, 1], color=c, label=label, alpha=0.8)
            ax.scatter(traj[0, 0], traj[0, 1], color="g", s=50, edgecolors="k")

    ax.set_title("Behavior Cloning Rollouts (Exact vs Perturbed Initial Conditions)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()
