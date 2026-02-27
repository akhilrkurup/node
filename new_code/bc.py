import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. Data Parsing & Velocity Extraction
# ==========================================
def load_and_preprocess_bc_data(file_names, nsamples=300):
    """
    Parses trajectories from .npy files and computes empirical velocities.
    Scaling has been removed; raw spatial values are used.
    """
    traj_all = []

    for file_name in file_names:
        # Load the numpy file
        traj = np.load(file_name)

        # Basic check to ensure it's a 2D array (points, features)
        if len(traj.shape) == 2:
            traj_all.append(traj)
        else:
            print(f"Warning: Skipping {file_name} due to shape {traj.shape}")

    total_trajs = len(traj_all)
    if total_trajs == 0:
        raise ValueError("No trajectories parsed. Check file paths and .npy shapes.")

    # Determine dimensions (e.g., 3 for your wiping data)
    dim = traj_all[0].shape[1]
    print(f"Successfully loaded {total_trajs} trajectories with dimension: {dim}")

    xs_process = np.zeros((total_trajs, nsamples, dim))
    ts_new = np.linspace(0, 1, nsamples)
    dt = ts_new[1] - ts_new[0]

    # Interpolate states to a uniform number of samples
    for i in range(total_trajs):
        # Create a time vector based on the original number of points in this specific file
        ts_original = np.linspace(0, 1, num=traj_all[i].shape[0])

        for j in range(dim):
            # Interpolate raw data directly without scaling
            f_interp = interpolate.interp1d(ts_original, traj_all[i][:, j])
            xs_process[i, :, j] = f_interp(ts_new)

    # Compute empirical velocities (dx/dt) from raw interpolated positions
    dxs_process = np.gradient(xs_process, axis=1) / dt

    xs_tensor = torch.tensor(xs_process, dtype=torch.float32)
    dxs_tensor = torch.tensor(dxs_process, dtype=torch.float32)

    return xs_tensor, dxs_tensor, dt


# ==========================================
# 2. BC Policy Architecture
# ==========================================
class BCPolicy(nn.Module):
    """Directly maps state x to velocity dx/dt"""

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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


# ==========================================
# 3. Rollout Simulation & Visualization
# ==========================================
def simulate_bc_rollout(model, x0, nsamples, dt):
    """Iteratively integrates the learned policy using the Euler method."""
    model.eval()
    traj_pred = [x0]
    x_curr = x0

    with torch.no_grad():
        for _ in range(nsamples - 1):
            v_pred = model(x_curr)
            # Euler integration step
            x_curr = x_curr + v_pred * dt
            traj_pred.append(x_curr)

    model.train()
    return torch.stack(traj_pred, dim=1)


def save_model(model, iter, output_dir="bc_policy"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = os.path.join(output_dir, f"bc_polic_iter{iter}.pth")
    torch.save(model.state_dict(), name)
    print(f"Saved model to {name}")


def plot_rollout(xs_true, xs_pred, step, output_dir="bc_rollouts"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xs_true_np = xs_true.detach().cpu().numpy()
    xs_pred_np = xs_pred.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(xs_true_np.shape[0]):
        plt.plot(
            xs_true_np[i, :, 0],
            xs_true_np[i, :, 1],
            "k--",
            alpha=0.5,
            label="Demonstrations" if i == 0 else "",
        )
        plt.plot(
            xs_pred_np[i, :, 0],
            xs_pred_np[i, :, 1],
            "r-",
            alpha=0.8,
            label="BC Euler Rollout" if i == 0 else "",
        )
        plt.scatter(
            xs_true_np[i, 0, 0],
            xs_true_np[i, 0, 1],
            c="blue",
            marker="x",
            s=50,
            zorder=5,
        )

    plt.title(f"Behavior Cloning Rollout - Iteration {step}")
    plt.xlabel("State $x_1$")
    plt.ylabel("State $x_2$")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"bc_rollout_step_{step:05d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ==========================================
# 4. Model Training
# ==========================================
def train_bc_model(xs, dxs, dt, steps=50000, lr=1e-4):
    """
    Solves the supervised learning problem mapping x -> dx.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    xs = xs.to(device)
    dxs = dxs.to(device)
    batch_size, nsamples, data_size = xs.shape

    # Flatten the temporal dimension to treat all states as independent samples
    xs_flat = xs.view(-1, data_size)
    dxs_flat = dxs.view(-1, data_size)

    # Initialize the model and optimizer
    model = BCPolicy(data_size, width_size=64, depth=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- ADD COSINE ANNEALING SCHEDULER ---
    # Decays from initial 'lr' down to 1% of the initial 'lr' over the total steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.01
    )

    criterion = nn.MSELoss()

    for step in range(steps):
        start = time.time()

        optimizer.zero_grad()

        # Predict velocities for all states simultaneously
        v_pred = model(xs_flat)

        # Loss is computed against the empirical target velocities
        loss = criterion(v_pred, dxs_flat)

        loss.backward()
        optimizer.step()

        # --- STEP THE SCHEDULER ---
        scheduler.step()

        end = time.time()

        if (step % 100) == 0 or step == steps - 1:
            # Fetch the current learning rate for logging
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step: {step:05d} | BC Loss: {loss.item():.6f} | LR: {current_lr:.6f} | Time: {end - start:.4f}s"
            )

            # Execute physical rollout from initial states to check for compounding errors
            x0 = xs[:, 0, :]
            xs_pred = simulate_bc_rollout(model, x0, nsamples, dt)
            plot_rollout(xs, xs_pred, step)
            save_model(model, step)
    return model


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":

    trajectory_files = [f"./wiping_data/wprr_demo{i}.npy" for i in range(1, 6)]

    print("Loading data and extracting empirical velocities...")
    xs, dxs, dt = load_and_preprocess_bc_data(trajectory_files)
    print(f"State tensor shape: {xs.shape}")

    print("\nInitializing Behavior Cloning optimization...")
    trained_bc_model = train_bc_model(xs, dxs, dt, steps=10000, lr=1e-4)

    final_out = "trained_bc_policy_final.pth"
    torch.save(trained_bc_model.state_dict(), final_out)
    print(f"\nFinal BC model state parameters serialized to {final_out}.")
