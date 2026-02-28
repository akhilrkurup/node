import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_rollout(file_path):
    # Load the saved rollout
    data = np.load(file_path)[0]
    print(f"Loaded rollout: {data.shape}")

    n_points, dim = data.shape

    # Calculate step-wise distance to represent velocity (for coloring)
    velocities = np.linalg.norm(np.diff(data, axis=0), axis=1)
    velocities = np.insert(velocities, 0, velocities[0])  # Match length

    fig = plt.figure(figsize=(10, 8))

    if dim == 3:
        # 3D Plotting
        ax = fig.add_subplot(111, projection="3d")
        # Use a scatter plot with a color map to show the flow
        p = ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            c=np.arange(n_points),
            cmap="viridis",
            s=10,
        )

        # Plot the line connecting them
        ax.plot(data[:, 0], data[:, 1], data[:, 2], "k-", alpha=0.3)

        # Mark Start (Green) and End (Red)
        ax.scatter(data[0, 0], data[0, 1], data[0, 2], c="g", s=100, label="Start")
        ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2], c="r", s=100, label="End")

        ax.set_zlabel("Z axis")
        fig.colorbar(p, label="Time Step (Flow Direction)")

    else:
        # 2D Plotting
        ax = fig.add_subplot(111)
        p = ax.scatter(
            data[:, 0], data[:, 1], c=np.arange(n_points), cmap="plasma", s=15
        )
        ax.plot(data[:, 0], data[:, 1], "k-", alpha=0.3)

        ax.scatter(data[0, 0], data[0, 1], c="g", s=100, label="Start")
        ax.scatter(data[-1, 0], data[-1, 1], c="r", s=100, label="End")
        fig.colorbar(p, label="Time Step")

    ax.set_title(f"Visualizing Rollout: {file_path}")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = r"C:\Users\akhil\OneDrive\Documents\GitHub\CLF-CBF-NODE\IROS\wiping\raw\node_rollouts_51.npy"
    visualize_rollout(file_path)
