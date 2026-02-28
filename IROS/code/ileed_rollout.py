import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==========================================
# 1. Define Architecture (Must match training)
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ILEED_Policy(nn.Module):
    def __init__(self, data_size, hidden_size, embed_dim, num_experts):
        super(ILEED_Policy, self).__init__()
        self.action_net = MLP(data_size, hidden_size, data_size)
        self.state_featurizer = MLP(data_size, hidden_size, embed_dim)
        self.omega = nn.Parameter(torch.randn(num_experts, embed_dim))

    def forward(self, x):
        # Rollout exclusively uses the pure distilled optimal policy
        return self.action_net(x)


# ==========================================
# 2. Simulation & Sampling Helpers
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


def simulate_trajectory(model, start_point, n_steps, dt):
    model.eval()
    current_state = torch.tensor(start_point, dtype=torch.float32)
    trajectory = [current_state.numpy()]

    with torch.no_grad():
        for _ in range(n_steps):
            velocity = model(current_state)
            current_state = current_state + velocity * dt
            trajectory.append(current_state.numpy())

    return np.array(trajectory)


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILEED Inference Rollout")
    parser.add_argument("--master_folder", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()

    master_folder = args.master_folder
    n_steps = args.n_steps
    dt = args.dt

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

    # ILEED requires the number of experts to match the shape of the embeddings during initialization
    NUM_EXPERTS = len(trajectory_files)
    WIDTH = 64
    EMBED_DIM = 4

    MODEL_PATH = os.path.join(master_folder, "ileed_policy/ileed_policy_iter02000.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ILEED_Policy(DATA_DIM, WIDTH, EMBED_DIM, NUM_EXPERTS).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.to("cpu")

    perturbation_radius = 0.001
    sampled_starts = sample_from_ball(
        exact_start_pos, perturbation_radius, num_samples=50, dim=DATA_DIM
    )
    all_starts = [exact_start_pos] + sampled_starts

    print(f"Simulating 51 ILEED rollouts from exact start {exact_start_pos}...")
    all_trajectories = []
    for pt in all_starts:
        traj = simulate_trajectory(model, pt, n_steps=n_steps, dt=dt)
        all_trajectories.append(traj)

    all_trajectories_np = np.array(all_trajectories)

    save_path = os.path.join(master_folder, "ileed_rollouts_51.npy")
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

    ax.set_title("ILEED Integration Rollouts (Exact vs Perturbed)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()
