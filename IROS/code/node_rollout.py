import argparse
import os
import random
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        solution = odeint(self.func, y0, ts, rtol=1e-3, atol=1e-4, method="dopri5")
        return solution.transpose(0, 1)


# ==========================================
# 2. Simulation Helpers
# ==========================================
def sample_from_ball(center, radius, num_samples, dim):
    points = []
    for _ in range(num_samples):
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)
        u = np.random.rand()
        r = radius * (u ** (1.0 / dim))
        points.append(center + r * direction)
    return points


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural ODE Inference Rollout")
    parser.add_argument("--master_folder", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Included for API consistency but bypassed by NODE's dopri5 solver.",
    )
    args = parser.parse_args()

    master_folder = args.master_folder
    n_steps = args.n_steps

    trajectory_files = [
        os.path.join(master_folder, f)
        for f in os.listdir(master_folder)
        if f.endswith(".npy") and "rollout" not in f
    ]
    trajectory_files.sort()

    if not trajectory_files:
        raise ValueError(f"No valid .npy demonstration files found in {master_folder}")

    # Dynamically extract start pos and dimensions
    first_demo = np.load(trajectory_files[0])
    exact_start_pos = first_demo[0, :]
    DATA_DIM = exact_start_pos.shape[0]

    WIDTH = 64
    DEPTH = 3
    MODEL_PATH = os.path.join(master_folder, "trained_node_policy_final.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralODE(DATA_DIM, WIDTH, DEPTH).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    perturbation_radius = 0.001
    sampled_starts = sample_from_ball(
        exact_start_pos, perturbation_radius, num_samples=50, dim=DATA_DIM
    )
    all_starts_np = np.vstack([exact_start_pos] + sampled_starts)
    start_pos_tensor = torch.tensor(all_starts_np, dtype=torch.float32).to(device)

    # Normalize the time tensor over the integration horizon
    ts = torch.linspace(0, 1, steps=n_steps).to(device)

    print(
        f"Integrating continuous vector field for 51 rollouts from exact start {exact_start_pos}..."
    )
    with torch.no_grad():
        generated_tensor = model(ts, start_pos_tensor)

    all_trajectories_np = generated_tensor.cpu().numpy()

    save_path = os.path.join(master_folder, "node_rollouts_51.npy")
    np.save(save_path, all_trajectories_np)
    print(f"Saved 51 rollouts to {save_path}")

    # ==========================================
    # 4. Visualization
    # ==========================================
    fig = plt.figure(figsize=(10, 7))
    plot_indices = [0] + random.sample(range(1, 51), 3)
    colors = ["r", "b", "c", "m"]
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
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()
