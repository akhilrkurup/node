import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_multiple_rollouts(file_paths, dim_x=0, dim_y=1, dim_z=2, is_3d=False):
    fig = plt.figure(figsize=(10, 10))

    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    valid_files_plotted = 0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ Error: File '{file_path}' not found. Skipping.")
            continue

        # Load the data
        data = np.load(file_path)

        # Check shape: Expected (num_rollouts, n_steps, dim)
        if len(data.shape) != 3:
            print(
                f"❌ Error: Expected 3D array of shape (num_rollouts, n_steps, dim) for '{file_path}', got {data.shape}"
            )
            continue

        num_rollouts, n_steps, dim = data.shape
        print(f"Loaded '{os.path.basename(file_path)}' ({n_steps} steps, {dim}D space)")

        if is_3d and dim <= max(dim_x, dim_y, dim_z):
            print(
                f"❌ Error: Cannot plot dims {dim_x}, {dim_y}, and {dim_z} for a {dim}D trajectory."
            )
            continue
        elif not is_3d and dim <= max(dim_x, dim_y):
            print(
                f"❌ Error: Cannot plot dims {dim_x} and {dim_y} for a {dim}D trajectory."
            )
            continue

        # Extract the exact first rollout (index 0)
        exact_rollout = data[0]
        label = os.path.dirname(file_path)

        if is_3d:
            # Plot the 3D trajectory line
            line = ax.plot(
                exact_rollout[:, dim_x],
                exact_rollout[:, dim_y],
                exact_rollout[:, dim_z],
                alpha=0.9,
                linewidth=2.5,
                label=label,
            )

            # Mark the 3D starting point
            color = line[0].get_color()
            ax.scatter(
                exact_rollout[0, dim_x],
                exact_rollout[0, dim_y],
                exact_rollout[0, dim_z],
                color=color,
                s=100,
                edgecolors="black",
                zorder=5,
            )
        else:
            # Plot the 2D trajectory line
            line = ax.plot(
                exact_rollout[:, dim_x],
                exact_rollout[:, dim_y],
                alpha=0.9,
                linewidth=2.5,
                label=label,
            )

            # Mark the 2D starting point
            color = line[0].get_color()
            ax.scatter(
                exact_rollout[0, dim_x],
                exact_rollout[0, dim_y],
                color=color,
                s=100,
                edgecolors="black",
                zorder=5,
            )

        valid_files_plotted += 1

    if valid_files_plotted == 0:
        print("❌ No valid files were plotted.")
        return

    # Titles and labels
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")

    if is_3d:
        ax.set_title(
            f"Comparison of Exact Start Rollouts (Dims {dim_x}, {dim_y}, {dim_z})"
        )
        ax.set_zlabel(f"Dimension {dim_z}")
    else:
        ax.set_title(f"Comparison of Exact Start Rollouts (Dims {dim_x} vs {dim_y})")
        # Force the scale of both axes to be strictly equal only in 2D to avoid projection distortion
        ax.set_aspect("equal", adjustable="box")

    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the first rollout of multiple .npy files in 2D or 3D."
    )

    # 'nargs="+"' allows you to pass multiple files separated by spaces
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Paths to the .npy rollouts files (e.g., file1.npy file2.npy)",
    )

    # Optional arguments to change which dimensions are plotted
    parser.add_argument(
        "--dim_x",
        type=int,
        default=0,
        help="Index of the dimension to plot on X axis (default: 0)",
    )
    parser.add_argument(
        "--dim_y",
        type=int,
        default=1,
        help="Index of the dimension to plot on Y axis (default: 1)",
    )
    parser.add_argument(
        "--dim_z",
        type=int,
        default=2,
        help="Index of the dimension to plot on Z axis in 3D mode (default: 2)",
    )

    # 3D visualization toggle
    parser.add_argument(
        "--plot_3d",
        action="store_true",
        help="Include this flag to plot the trajectories in full 3D",
    )

    args = parser.parse_args()
    visualize_multiple_rollouts(
        args.files, args.dim_x, args.dim_y, args.dim_z, args.plot_3d
    )
