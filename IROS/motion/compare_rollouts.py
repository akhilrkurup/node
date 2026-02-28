import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_multiple_rollouts(file_paths, dim_x=0, dim_y=1):
    fig, ax = plt.subplots(figsize=(10, 10))
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

        if dim <= max(dim_x, dim_y):
            print(
                f"❌ Error: Cannot plot dims {dim_x} and {dim_y} for a {dim}D trajectory."
            )
            continue

        # Extract the exact first rollout (index 0)
        exact_rollout = data[0]
        label = os.path.dirname(file_path)

        # Plot the trajectory line
        line = ax.plot(
            exact_rollout[:, dim_x],
            exact_rollout[:, dim_y],
            alpha=0.9,
            linewidth=2.5,
            label=label,
        )

        # Mark the starting point, matching the color of the trajectory line
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

    # Force the scale of both axes to be strictly equal
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"Comparison of Exact Start Rollouts (Dims {dim_x} vs {dim_y})")
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the first rollout of multiple .npy files in 2D."
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

    args = parser.parse_args()
    visualize_multiple_rollouts(args.files, args.dim_x, args.dim_y)
