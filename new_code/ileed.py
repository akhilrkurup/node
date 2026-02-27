import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import time


# ==========================================
# 1. Data Preprocessing (Retaining Expert IDs)
# ==========================================
def load_and_preprocess_ileed_data(file_names, nsamples=300):
    """
    Parses trajectories from .npy files and computes empirical velocities.
    Crucially, it assigns a 'trajectory ID' to each state so ILEED can
    learn the specific expertise (omega) of that demonstration.
    """
    traj_all = []

    for file_name in file_names:
        traj = np.load(file_name)
        if len(traj.shape) == 2:
            traj_all.append(traj)

    num_experts = len(traj_all)
    if num_experts == 0:
        raise ValueError("No trajectories parsed.")

    dim = traj_all[0].shape[1]

    xs_process = np.zeros((num_experts, nsamples, dim))
    ts_new = np.linspace(0, 1, nsamples)
    dt = ts_new[1] - ts_new[0]

    for i in range(num_experts):
        ts_original = np.linspace(0, 1, num=traj_all[i].shape[0])
        for j in range(dim):
            f_interp = interpolate.interp1d(ts_original, traj_all[i][:, j])
            xs_process[i, :, j] = f_interp(ts_new)

    dxs_process = np.gradient(xs_process, axis=1) / dt

    # Create an array of Expert IDs matching the state shapes
    # Shape: (num_experts, nsamples)
    expert_ids = np.repeat(np.arange(num_experts)[:, None], nsamples, axis=1)

    xs_tensor = torch.tensor(xs_process, dtype=torch.float32)
    dxs_tensor = torch.tensor(dxs_process, dtype=torch.float32)
    ids_tensor = torch.tensor(expert_ids, dtype=torch.long)

    return xs_tensor, dxs_tensor, ids_tensor, dt, num_experts


# ==========================================
# 2. Continuous ILEED Architecture
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

        # 1. Action Network (The policy we actually want to extract)
        self.action_net = MLP(data_size, hidden_size, data_size)

        # 2. State Featurizer (Determines how 'difficult' a state is)
        self.state_featurizer = MLP(data_size, hidden_size, embed_dim)

        # 3. Expert Embeddings (Learnable proficiency profiles per trajectory)
        self.omega = nn.Parameter(torch.randn(num_experts, embed_dim))

    def forward_training(self, x, expert_ids):
        """Used during training to compute action and competence score."""
        v_pred = self.action_net(x)
        state_features = self.state_featurizer(x)

        # Extract the specific proficiency vectors for the batch
        batch_omega = self.omega[expert_ids]

        # Compute state-dependent competence sigma = Sigmoid(f(s) * omega)
        dot_product = torch.sum(state_features * batch_omega, dim=-1)
        sigma = torch.sigmoid(dot_product)

        return v_pred, sigma

    def forward(self, x):
        """Used during rollout. We assume optimal competence (sigma=1) and just use the action network."""
        return self.action_net(x)


# ==========================================
# 3. Continuous IRT Mixture Loss
# ==========================================
def continuous_irt_loss(v_pred, v_true, sigma, tau=0.1, random_floor=0.01):
    """
    Continuous adaptation of the ILEED IRT loss.
    tau: Controls the sharpness of the optimal Gaussian policy.
    random_floor: The likelihood of an action if the expert is acting irrationally.
    """
    # Euclidean squared distance between predicted and demonstrated velocity
    mse = torch.sum((v_pred - v_true) ** 2, dim=-1)

    # Probability density of the optimal policy (Un-normalized Gaussian)
    p_opt = torch.exp(-mse / tau)

    # Mixture Model Likelihood
    likelihood = (sigma * p_opt) + ((1.0 - sigma) * random_floor)

    # Negative Log Likelihood
    nll = -torch.log(likelihood + 1e-8)
    return nll.mean()


# ==========================================
# 4. Training Loop
# ==========================================
def train_ileed(xs, dxs, ids, num_experts, dt, steps=10000, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Continuous ILEED on device: {device}")

    xs = xs.to(device)
    dxs = dxs.to(device)
    ids = ids.to(device)
    batch_size, nsamples, data_size = xs.shape

    # Flatten for I.I.D processing
    xs_flat = xs.view(-1, data_size)
    dxs_flat = dxs.view(-1, data_size)
    ids_flat = ids.view(-1)

    model = ILEED_Policy(
        data_size=data_size, hidden_size=64, embed_dim=4, num_experts=num_experts
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )

    for step in range(steps):
        start = time.time()
        optimizer.zero_grad()

        # Forward pass calculates both the velocity and the competence (sigma)
        v_pred, sigma = model.forward_training(xs_flat, ids_flat)

        # Calculate continuous IRT loss
        loss = continuous_irt_loss(v_pred, dxs_flat, sigma, tau=0.5, random_floor=0.05)

        loss.backward()
        optimizer.step()
        scheduler.step()

        end = time.time()

        if (step % 500) == 0 or step == steps - 1:
            print(
                f"Step: {step:05d} | IRT Loss: {loss.item():.4f} | Mean Sigma (Competence): {sigma.mean().item():.3f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return model


# ==========================================
# 5. Rollout / Inference Function
# ==========================================
def simulate_ileed_rollout(model, start_point, n_steps=1000, dt=0.01):
    """
    Rolls out a trajectory using Euler integration.
    Notice we only query `model(x)`, dropping the featurizer and omega,
    because we want the robot to execute strictly optimal behavior.
    """
    model.eval()
    device = next(model.parameters()).device

    current_state = torch.tensor(start_point, dtype=torch.float32).to(device)
    trajectory = [current_state.detach().cpu().numpy()]

    with torch.no_grad():
        for _ in range(n_steps):
            velocity = model(current_state)  # Queries action_net directly
            current_state = current_state + velocity * dt
            trajectory.append(current_state.detach().cpu().numpy())

    return np.array(trajectory)


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":

    # Replace with your actual unscaled npy paths
    trajectory_files = [
        r"./wiping_data/wprr_demo1.npy",
        r"./wiping_data/wprr_demo2.npy",
        r"./wiping_data/wprr_demo3.npy",
        r"./wiping_data/wprr_demo4.npy",
        r"./wiping_data/wprr_demo5.npy",
    ]

    print("Loading data and extracting empirical velocities + expert IDs...")
    xs, dxs, expert_ids, dt, num_experts = load_and_preprocess_ileed_data(
        trajectory_files
    )
    print(f"State tensor shape: {xs.shape}")

    print("\nInitializing Continuous ILEED optimization...")
    trained_ileed_model = train_ileed(
        xs, dxs, expert_ids, num_experts, dt, steps=8000, lr=1e-3
    )

    # Save the architecture
    final_out = "trained_ileed_policy.pth"
    torch.save(trained_ileed_model.state_dict(), final_out)
    print(f"\nFinal ILEED model state serialized to {final_out}.")

    # Example Rollout
    start_pos = xs[
        0, 0, :
    ].numpy()  # Use the first point of the first demo as a test start
    rollout_traj = simulate_ileed_rollout(
        trained_ileed_model, start_pos, n_steps=300, dt=dt
    )
    np.save("ileed_rollout.npy", rollout_traj)
    print("Test rollout saved to ileed_rollout.npy.")
