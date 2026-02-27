import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import pickle


def load_and_preprocess_data(file_names, nsamples=300):
    """
    Parses trajectories from .npy files and interpolates them to a uniform
    length using raw spatial coordinates (no scaling applied).
    """
    traj_all = []

    for file_name in file_names:
        # Load the numpy file
        traj = np.load(file_name)
        if len(traj.shape) == 2:
            traj_all.append(traj)
        else:
            print(f"Warning: Skipping {file_name} due to shape {traj.shape}")

    total_trajs = len(traj_all)
    if total_trajs == 0:
        raise ValueError("No trajectories parsed. Verify file paths.")

    dim = traj_all[0].shape[1]

    # Pre-allocate batched trajectory tensor
    traj_all_process = np.zeros((total_trajs, nsamples, dim))
    ts_new = np.linspace(0, 1, nsamples)

    # 1D interpolation over raw states to uniformize the sequence length
    for i in range(total_trajs):
        # Create a time vector based on the original point count for this trajectory
        ts_original = np.linspace(0, 1, num=traj_all[i].shape[0])

        for j in range(dim):
            # Interpolating raw data directly
            f_interp = interpolate.interp1d(ts_original, traj_all[i][:, j])
            traj_all_process[i, :, j] = f_interp(ts_new)

    # Convert to PyTorch tensors
    ts_tensor = torch.tensor(ts_new, dtype=torch.float32)
    ys_tensor = torch.tensor(traj_all_process, dtype=torch.float32)

    print(f"Successfully processed {total_trajs} raw trajectories.")
    return ts_tensor, ys_tensor


# ==========================================
# 2. Neural ODE Architecture
# ==========================================
class VectorField(nn.Module):
    """Parameterizes the vector field: dx/dt = f(x, t)"""

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

        # Initialize with orthogonal weights to improve continuous depth flow
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t, y):
        # torchdiffeq requires the signature forward(t, y)
        return self.mlp(y)


class NeuralODE(nn.Module):
    """Wrapper that solves the Initial Value Problem using numerical integration."""

    def __init__(self, data_size, width_size, depth):
        super(NeuralODE, self).__init__()
        self.func = VectorField(data_size, width_size, depth)

    def forward(self, ts, y0):
        # Solve ODE using dopri5 (5th order Runge-Kutta)
        solution = odeint(self.func, y0, ts, rtol=1e-3, atol=1e-6, method="dopri5")
        # Transpose to return [batch_size, time_steps, data_size]
        return solution.transpose(0, 1)


# ==========================================
# 3. Rollout Visualization
# ==========================================
def plot_rollout(ys_true, ys_pred, step, output_dir="rollouts"):
    """
    Projects the integrated trajectory states and ground truth onto a 2D plane
    to visually evaluate limit cycle convergence and tracking fidelity.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ys_true_np = ys_true.detach().cpu().numpy()
    ys_pred_np = ys_pred.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(ys_true_np.shape[0]):
        plt.plot(
            ys_true_np[i, :, 0],
            ys_true_np[i, :, 1],
            "k--",
            alpha=0.5,
            label="Demonstrations" if i == 0 else "",
        )
        plt.plot(
            ys_pred_np[i, :, 0],
            ys_pred_np[i, :, 1],
            "g-",
            alpha=0.8,
            label="NODE Integrated Flow" if i == 0 else "",
        )

        # Mark initial conditions
        plt.scatter(
            ys_true_np[i, 0, 0],
            ys_true_np[i, 0, 1],
            c="red",
            marker="x",
            s=50,
            zorder=5,
        )

    plt.title(f"Dynamical System Rollout - Iteration {step}")
    plt.xlabel("State $x_1$")
    plt.ylabel("State $x_2$")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"rollout_step_{step:04d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ==========================================
# 4. Model Training
# ==========================================
def train_model(ts, ys, steps=3000, lr=3e-3, load_path=None):
    """
    Solves the empirical risk minimization problem mapping x_0 -> x_{0:T}
    using the Adam optimizer and Cosine Annealing learning rate schedule.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Training on device: {device}")

    ts = ts.to(device)
    ys = ys.to(device)

    batch_size, length_size, data_size = ys.shape

    # Initialize the model architecture
    model = NeuralODE(data_size, width_size=64, depth=3).to(device)

    # --- WEIGHT INITIALIZATION INJECTION ---
    if load_path is not None and os.path.exists(load_path):
        print(f"Loading pre-trained parameter manifold from: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print(
            "No valid checkpoint found. Initializing orthogonal weights from scratch..."
        )
    # ---------------------------------------

    # Configure optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    criterion = nn.MSELoss()

    # Execution Loop
    for step in range(steps):
        start = time.time()

        optimizer.zero_grad()

        # Extract initial conditions x(t=0) for the whole batch
        y0 = ys[:, 0, :]

        # Forward pass (integration via RK45)
        y_pred = model(ts, y0)

        # Compute trajectory reconstruction MSE
        loss = criterion(y_pred, ys)

        # Backpropagation
        loss.backward()
        optimizer.step()
        # scheduler.step()

        end = time.time()

        # Execute rollout evaluation and logging
        if (step % 100) == 0 or step == steps - 1:
            print(
                f"Step: {step:04d} | Loss: {loss.item():.6f} | Computation time: {end - start:.4f}s"
            )
            plot_rollout(ys, y_pred, step)

    return model


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":

    # Define an iterable list of trajectory paths
    trajectory_files = [f"./wiping_data/wprr_demo{i}.npy" for i in range(1, 6)]

    print("Loading and aggregating multidimensional trajectories...")
    ts, ys = load_and_preprocess_data(trajectory_files)
    print(f"Aggregated tensor shape (batch, time, dim): {ys.shape}")

    print("\nInitializing Neural ODE optimization sequence...")
    trained_model = train_model(ts, ys, steps=10000)

    # Save the PyTorch model state dictionary
    out_file = "trained_node_policy.pth"
    torch.save(trained_model.state_dict(), out_file)
    print(f"\nModel state parameters serialized to {out_file}.")
