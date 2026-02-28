import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os


# ==========================================
# 1. Define Architecture (Must match training)
# ==========================================
class VectorField(nn.Module):
    def __init__(self, data_size, width_size, depth):
        super(VectorField, self).__init__()
        layers = []
        in_features = data_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width_size))
            layers.append(nn.Tanh())
            in_features = width_size
        layers.append(nn.Linear(in_features, data_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, y):
        # Time-invariant autonomous system; 't' is unused but required by odeint
        return self.mlp(y)


class NeuralODE(nn.Module):
    def __init__(self, data_size, width_size, depth):
        super(NeuralODE, self).__init__()
        self.func = VectorField(data_size, width_size, depth)

    def forward(self, ts, y0):
        # We use a relaxed tolerance for faster inference
        solution = odeint(self.func, y0, ts, rtol=1e-3, atol=1e-4, method="dopri5")
        # Transpose to [batch, time, dim]
        return solution.transpose(0, 1)


# ==========================================
# 2. Simulation Helpers
# ==========================================
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
    # Parameters (Must match how you trained the .pth file)
    DATA_DIM = 3  # Change to 2 if testing the 2D drawing data
    WIDTH = 64
    DEPTH = 3
    master_folder = "./IROS/wiping/raw"
    MODEL_PATH = os.path.join(master_folder, "trained_node_policy_final.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralODE(DATA_DIM, WIDTH, DEPTH).to(device)

    # Load weights safely with weights_only=False to prevent pickling errors
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # Define the exact starting condition
    exact_start_pos = np.array([0.65, 0.0, 0.0])

    # Generate 50 additional starting points within a 0.1 radius ball
    perturbation_radius = 0.1
    sampled_starts = sample_from_ball(
        exact_start_pos, perturbation_radius, num_samples=50, dim=DATA_DIM
    )

    # Combine into a single array of shape (51, 3)
    all_starts_np = np.vstack([exact_start_pos] + sampled_starts)

    # Convert to a batched PyTorch tensor
    start_pos_tensor = torch.tensor(all_starts_np, dtype=torch.float32).to(device)

    # Define the integration time horizon
    n_steps = 300
    ts = torch.linspace(0, 1, steps=n_steps).to(device)

    print(f"Integrating continuous vector field for 51 rollouts simultaneously...")
    with torch.no_grad():
        # Because we pass a batched tensor of initial conditions,
        # odeint integrates all 51 trajectories at the exact same time.
        generated_tensor = model(ts, start_pos_tensor)

    # Move back to CPU and convert to numpy. Shape: (51, 300, 3)
    all_trajectories_np = generated_tensor.cpu().numpy()

    # Save all trajectories
    save_path = os.path.join(master_folder, "node_rollouts_51.npy")
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

    if DATA_DIM == 3:
        ax = fig.add_subplot(111, projection="3d")
        for idx, c, label in zip(plot_indices, colors, labels):
            traj = all_trajectories_np[idx]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, label=label, alpha=0.8)
            ax.scatter(
                traj[0, 0], traj[0, 1], traj[0, 2], color="g", s=50, edgecolors="k"
            )

        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        for idx, c, label in zip(plot_indices, colors, labels):
            traj = all_trajectories_np[idx]
            ax.plot(traj[:, 0], traj[:, 1], color=c, label=label, alpha=0.8)
            ax.scatter(traj[0, 0], traj[0, 1], color="g", s=50, edgecolors="k")

    ax.set_title("Neural ODE Integration Rollouts (Exact vs Perturbed)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()
