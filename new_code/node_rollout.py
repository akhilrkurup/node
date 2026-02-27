import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


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
# 2. Main Execution
# ==========================================
if __name__ == "__main__":
    # Parameters (Must match how you trained the .pth file)
    DATA_DIM = 3  # Change to 2 if testing the 2D drawing data
    WIDTH = 64
    DEPTH = 3
    MODEL_PATH = "./wiping_data/trained_node_policy_new.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralODE(DATA_DIM, WIDTH, DEPTH).to(device)

    # Load weights safely handling both dict formats
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # Define initial starting condition (Must be in your normalized range)
    # Shape must be [batch_size, data_size], e.g., [1, 3]
    start_pos = torch.tensor([[0.65, 0, 0.0]], dtype=torch.float32).to(device)

    # Define the integration time horizon
    # t=0 to t=1 corresponds exactly to one demonstrated loop based on your scaler
    # Increasing n_steps just increases the resolution of the returned points
    n_steps = 300
    ts = torch.linspace(0, 1, steps=n_steps).to(device)

    print("Integrating continuous vector field...")
    with torch.no_grad():
        # Execute the forward pass through the ODE solver
        generated_tensor = model(ts, start_pos)

    # Extract the first (and only) batch, convert to numpy
    generated_traj = generated_tensor[0].cpu().numpy()

    np.save("./wiping_data/node_rollout.npy", generated_traj)

    # ==========================================
    # 3. Visualization
    # ==========================================
    fig = plt.figure(figsize=(10, 7))
    if DATA_DIM == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            generated_traj[:, 0],
            generated_traj[:, 1],
            "g-",
            label="NODE Flow",
        )
        ax.scatter(
            generated_traj[0, 0],
            generated_traj[0, 1],
            c="red",
            s=100,
            label="Start",
        )
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        ax.plot(generated_traj[:, 0], generated_traj[:, 1], "g-", label="NODE Flow")
        ax.scatter(
            generated_traj[0, 0], generated_traj[0, 1], c="red", s=100, label="Start"
        )

    ax.set_title("Neural ODE Integration Rollout (One Period)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.legend()
    plt.show()
