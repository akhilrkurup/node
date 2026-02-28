import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_rollouts(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    # Load the data
    data = np.load(file_path)

    # Check shape: Expected (num_rollouts, n_steps, dim)
    if len(data.shape) != 3:
        print(
            f"❌ Error: Expected 3D array of shape (num_rollouts, n_steps, dim), got {data.shape}"
        )
        return

    num_rollouts, n_steps, dim = data.shape
    print(f"Loaded {num_rollouts} rollouts with {n_steps} steps in {dim}D space.")

    fig = plt.figure(figsize=(10, 7))

    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")

        # 1. Plot the perturbed rollouts in semi-transparent grey
        for i in range(1, num_rollouts):
            # Only add the label once to avoid cluttering the legend
            label = "Perturbed Rollouts" if i == 1 else ""
            ax.plot(
                data[i, :, 0],
                data[i, :, 1],
                data[i, :, 2],
                color="grey",
                alpha=0.3,
                linewidth=1,
                label=label,
            )

        # 2. Plot the exact start rollout in solid red on top
        ax.plot(
            data[0, :, 0],
            data[0, :, 1],
            data[0, :, 2],
            color="red",
            alpha=1.0,
            linewidth=2.5,
            label="Exact Start Rollout",
        )

        # Mark the starting point of the exact rollout
        ax.scatter(
            data[0, 0, 0],
            data[0, 0, 1],
            data[0, 0, 2],
            color="green",
            s=100,
            edgecolors="black",
            label="Start Point",
        )

        ax.set_zlabel("Z")

    elif dim == 2:
        ax = fig.add_subplot(111)

        # 1. Plot the perturbed rollouts in semi-transparent grey
        for i in range(1, num_rollouts):
            label = "Perturbed Rollouts" if i == 1 else ""
            ax.plot(
                data[i, :, 0],
                data[i, :, 1],
                color="grey",
                alpha=0.3,
                linewidth=1,
                label=label,
            )

        # 2. Plot the exact start rollout in solid red on top
        ax.plot(
            data[0, :, 0],
            data[0, :, 1],
            color="red",
            alpha=1.0,
            linewidth=2.5,
            label="Exact Start Rollout",
        )

        # Mark the starting point
        ax.scatter(
            data[0, 0, 0],
            data[0, 0, 1],
            color="green",
            s=100,
            edgecolors="black",
            label="Start Point",
            zorder=5,
        )

    else:
        print(f"❌ Error: Cannot plot data with dimensionality {dim}.")
        return

    ax.set_title(f"Rollout Phase Space: {os.path.basename(file_path)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved rollout .npy files.")
    parser.add_argument(
        "file",
        type=str,
        help="Path to the .npy rollouts file (e.g., ./IROS/wiping/raw/bc_rollouts_51.npy)",
    )

    args = parser.parse_args()
    visualize_rollouts(args.file)
