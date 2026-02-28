import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
import matplotlib.pyplot as plt


# ==========================================
# 1. Data Preprocessing (Retaining Expert IDs)
# ==========================================
def load_and_preprocess_ileed_data(file_names, nsamples=300):
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
        self.action_net = MLP(data_size, hidden_size, data_size)
        self.state_featurizer = MLP(data_size, hidden_size, embed_dim)
        self.omega = nn.Parameter(torch.randn(num_experts, embed_dim))

    def forward_training(self, x, expert_ids):
        v_pred = self.action_net(x)
        state_features = self.state_featurizer(x)
        batch_omega = self.omega[expert_ids]
        dot_product = torch.sum(state_features * batch_omega, dim=-1)
        sigma = torch.sigmoid(dot_product)
        return v_pred, sigma

    def forward(self, x):
        # Pure policy extraction
        return self.action_net(x)


# ==========================================
# 3. Continuous IRT Loss & Checkpointing
# ==========================================
def continuous_irt_loss(v_pred, v_true, sigma, tau=0.1, random_floor=0.01):
    mse = torch.sum((v_pred - v_true) ** 2, dim=-1)
    p_opt = torch.exp(-mse / tau)
    likelihood = (sigma * p_opt) + ((1.0 - sigma) * random_floor)
    nll = -torch.log(likelihood + 1e-8)
    return nll.mean()


def save_model(model, step, output_dir="ileed_policy"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = os.path.join(output_dir, f"ileed_policy_iter{step:05d}.pth")
    torch.save(model.state_dict(), name)


def simulate_ileed_rollout(model, start_point, n_steps=300, dt=0.01):
    model.eval()
    device = next(model.parameters()).device
    current_state = torch.tensor(start_point, dtype=torch.float32).to(device)
    trajectory = [current_state.detach().cpu().numpy()]

    with torch.no_grad():
        for _ in range(n_steps - 1):
            velocity = model(current_state)
            current_state = current_state + velocity * dt
            trajectory.append(current_state.detach().cpu().numpy())
    return np.array(trajectory)


def plot_rollout(xs_true, model, nsamples, dt, step, output_dir="ileed_rollouts"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xs_true_np = xs_true.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))

    for i in range(xs_true_np.shape[0]):
        # Plot Expert Ground Truth
        plt.plot(
            xs_true_np[i, :, 0],
            xs_true_np[i, :, 1],
            "k--",
            alpha=0.5,
            label="Demonstrations" if i == 0 else "",
        )

        # Simulate from this expert's initial condition
        x0 = xs_true_np[i, 0, :]
        traj_pred = simulate_ileed_rollout(model, x0, nsamples, dt)

        # Plot the distilled model's rollout
        plt.plot(
            traj_pred[:, 0],
            traj_pred[:, 1],
            "r-",
            alpha=0.8,
            label="ILEED Rollout" if i == 0 else "",
        )
        plt.scatter(x0[0], x0[1], c="blue", marker="x", s=50, zorder=5)

    plt.title(f"ILEED Policy Distillation - Iteration {step}")
    plt.xlabel("State $x_1$")
    plt.ylabel("State $x_2$")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f"ileed_rollout_step_{step:05d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ==========================================
# 4. Training Loop
# ==========================================
def train_ileed(xs, dxs, ids, num_experts, dt, steps=10000, lr=1e-3, output_base="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Continuous ILEED on device: {device}")

    xs = xs.to(device)
    dxs = dxs.to(device)
    ids = ids.to(device)
    batch_size, nsamples, data_size = xs.shape

    xs_flat = xs.view(-1, data_size)
    dxs_flat = dxs.view(-1, data_size)
    ids_flat = ids.view(-1)

    model = ILEED_Policy(data_size, 64, 4, num_experts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )

    model_dir = os.path.join(output_base, "ileed_policy")
    plot_dir = os.path.join(output_base, "ileed_rollouts")

    for step in range(steps):
        start = time.time()
        optimizer.zero_grad()

        v_pred, sigma = model.forward_training(xs_flat, ids_flat)
        loss = continuous_irt_loss(v_pred, dxs_flat, sigma, tau=0.5, random_floor=0.05)

        loss.backward()
        optimizer.step()
        scheduler.step()
        end = time.time()

        if (step % 100) == 0 or step == steps - 1:
            print(
                f"Step: {step:05d} | IRT Loss: {loss.item():.4f} | Mean Sigma: {sigma.mean().item():.3f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            save_model(model, step, output_dir=model_dir)
            plot_rollout(xs, model, nsamples, dt, step, output_dir=plot_dir)

    return model


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    master_folder = "./IROS/pick/processed"
    trajectory_files = [
        os.path.join(master_folder, f"pnp_demo{i}.npy") for i in range(1, 6)
    ]

    print("Loading data and extracting empirical velocities + expert IDs...")
    xs, dxs, expert_ids, dt, num_experts = load_and_preprocess_ileed_data(
        trajectory_files
    )
    print(f"State tensor shape: {xs.shape}, Total Experts: {num_experts}")

    trained_ileed_model = train_ileed(
        xs,
        dxs,
        expert_ids,
        num_experts,
        dt,
        steps=10000,
        lr=1e-3,
        output_base=master_folder,
    )

    final_out = os.path.join(master_folder, "trained_ileed_policy_final.pth")
    torch.save(trained_ileed_model.state_dict(), final_out)
    print(f"\nFinal ILEED model state serialized to {final_out}.")
