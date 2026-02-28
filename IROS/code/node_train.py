import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import argparse


# ==========================================
# 1. Data Parsing & Interpolation
# ==========================================
def load_and_preprocess_node_data(file_names, nsamples=300):
    """
    Parses trajectories from .npy files and interpolates them over a
    normalized time horizon t in [0, 1] for Neural ODE integration.
    Scaling has been removed; raw spatial values are used.
    """
    traj_all = []

    for file_name in file_names:
        traj = np.load(file_name)
        if len(traj.shape) == 2:
            traj_all.append(traj)
        else:
            print(f"Warning: Skipping {file_name} due to shape {traj.shape}")

    total_trajs = len(traj_all)
    if total_trajs == 0:
        raise ValueError("No trajectories parsed. Check file paths and .npy shapes.")

    dim = traj_all[0].shape[1]
    print(f"Successfully loaded {total_trajs} trajectories with dimension: {dim}")

    xs_process = np.zeros((total_trajs, nsamples, dim))
    ts_new = np.linspace(0, 1, nsamples)

    # Interpolate states to a uniform time grid
    for i in range(total_trajs):
        ts_original = np.linspace(0, 1, num=traj_all[i].shape[0])
        for j in range(dim):
            f_interp = interpolate.interp1d(ts_original, traj_all[i][:, j])
            xs_process[i, :, j] = f_interp(ts_new)

    xs_tensor = torch.tensor(xs_process, dtype=torch.float32)
    ts_tensor = torch.tensor(ts_new, dtype=torch.float32)

    return xs_tensor, ts_tensor


# ==========================================
# 2. Neural ODE Architecture
# ==========================================
class VectorField(nn.Module):
    def __init__(self, data_size, width_size, depth):
        super(VectorField, self).__init__()
        layers = []
        in_features = data_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width_size))
            layers.append(nn.Tanh())  # Swap to nn.ReLU() if raw values are too large
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
        # Forward pass integrates the vector field over the time tensor
        solution = odeint(self.func, y0, ts, rtol=1e-3, atol=1e-4, method="dopri5")
        # Transpose to [batch, time, dim] to match the training data shape
        return solution.transpose(0, 1)


# ==========================================
# 3. Checkpointing & Visualization
# ==========================================
def save_model(model, step, output_dir="node_policy"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = os.path.join(output_dir, f"node_policy_iter{step:05d}.pth")
    torch.save(model.state_dict(), name)


def plot_rollout(xs_true, xs_pred, step, output_dir="node_rollouts"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xs_true_np = xs_true.detach().cpu().numpy()
    xs_pred_np = xs_pred.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(xs_true_np.shape[0]):
        # Ground Truth
        plt.plot(
            xs_true_np[i, :, 0],
            xs_true_np[i, :, 1],
            "k--",
            alpha=0.5,
            label="Demonstrations" if i == 0 else "",
        )
        # ODE Integration Flow
        plt.plot(
            xs_pred_np[i, :, 0],
            xs_pred_np[i, :, 1],
            "g-",
            alpha=0.8,
            label="Neural ODE Rollout" if i == 0 else "",
        )
        # Starting point
        plt.scatter(
            xs_true_np[i, 0, 0],
            xs_true_np[i, 0, 1],
            c="blue",
            marker="x",
            s=50,
            zorder=5,
        )

    plt.title(f"Neural ODE Rollout - Iteration {step}")
    plt.xlabel("State $x_1$")
    plt.ylabel("State $x_2$")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"node_rollout_step_{step:05d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ==========================================
# 4. Model Training
# ==========================================
def train_node_model(xs, ts, steps=10000, lr=1e-3, output_base="."):
    """
    Trains the Neural ODE by matching the integrated trajectory to the demonstrations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Training NODE on device: {device}")

    xs = xs.to(device)
    ts = ts.to(device)
    batch_size, nsamples, data_size = xs.shape

    # Initial conditions for the ODE solver
    x0 = xs[:, 0, :]

    model = NeuralODE(data_size, width_size=64, depth=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01
    )

    criterion = nn.MSELoss()

    # Define the dynamic subfolder paths inside the master folder
    model_dir = os.path.join(output_base, "node_policy")
    plot_dir = os.path.join(output_base, "node_rollouts")

    for step in range(steps):
        start = time.time()
        optimizer.zero_grad()

        # Integrate from x0 over all time steps
        xs_pred = model(ts, x0)

        # Compute Loss over the entire trajectory timeline
        loss = criterion(xs_pred, xs)

        loss.backward()
        optimizer.step()
        scheduler.step()

        end = time.time()

        if (step % 100) == 0 or step == steps - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step: {step:05d} | NODE Traj Loss: {loss.item():.6f} | LR: {current_lr:.6f} | Time: {end - start:.4f}s"
            )

            # Save models and plot trajectories
            plot_rollout(xs, xs_pred, step, output_dir=plot_dir)
            save_model(model, step, output_dir=model_dir)

    return model


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(description="Train Neural ODE Policy")
    parser.add_argument(
        "--master_folder",
        type=str,
        required=True,
        help="Path to the master folder containing .npy demonstration files (e.g., ./IROS/wiping/raw)",
    )
    args = parser.parse_args()

    master_folder = args.master_folder

    # 2. Dynamically target all .npy files in the master directory
    # We filter out 'rollout' so it only trains on the demonstration data
    trajectory_files = [
        os.path.join(master_folder, f)
        for f in os.listdir(master_folder)
        if f.endswith(".npy") and "rollout" not in f
    ]

    # Sort files to ensure deterministic loading order
    trajectory_files.sort()

    if not trajectory_files:
        raise ValueError(f"No valid .npy demonstration files found in {master_folder}")

    print(f"Using files: {trajectory_files}")
    print("Loading data and interpolating over time...")
    xs, ts = load_and_preprocess_node_data(trajectory_files)
    print(f"State tensor shape: {xs.shape}")

    print("\nInitializing Neural ODE optimization...")
    # Pass the master_folder to route the outputs cleanly
    # (Make sure to set steps=10000 or whatever you need for the actual training!)
    trained_node_model = train_node_model(
        xs, ts, steps=10000, lr=1e-3, output_base=master_folder
    )

    # 3. Save the final model weights
    final_out = os.path.join(master_folder, "trained_node_policy_final.pth")
    torch.save(trained_node_model.state_dict(), final_out)
    print(f"\nFinal NODE parameters serialized to {final_out}.")
