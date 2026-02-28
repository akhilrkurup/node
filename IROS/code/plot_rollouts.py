import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_multiple_rollouts(npy_files):
    """
    Loads and plots multiple .npy trajectory files on the same figure.
    Dynamically supports 2D and 3D state spaces.
    """
    if not npy_files:
        print("No files provided in the list.")
        return

    # Extract the ambient dimension from the first file to set up the axes
    try:
        first_traj = np.load(npy_files[0])
        dim = first_traj.shape[1]
    except Exception as e:
        print(f"Failed to load the reference file {npy_files[0]}: {e}")
        return
    # Initialize Plot
    fig = plt.figure(figsize=(12, 9))
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("Z coordinate (m)")
    else:
        ax = fig.add_subplot(111)

    # Generate a distinct color palette for the requested number of trajectories
    # tab20 supports up to 20 highly distinct colors for dense comparisons
    colors = plt.cm.tab20(np.linspace(0, 1, len(npy_files)))

    successful_plots = 0

    for idx, file_path in enumerate(npy_files):
        try:
            # Load trajectory
            traj = np.load(file_path)
            name = os.path.basename(file_path)

            # Mathematical validation: Ensure all trajectories belong to the same manifold dimension
            if traj.shape[1] != dim:
                print(
                    f"Warning: Skipping '{name}'. Dimensionality mismatch (Expected {dim}, got {traj.shape[1]})."
                )
                continue

            # Plotting logic
            if dim == 3:
                # Plot continuous flow
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    color=colors[idx],
                    label=name,
                    linewidth=2,
                    alpha=0.8,
                )
                # Mark Start Condition (Circle)
                ax.scatter(
                    traj[0, 0],
                    traj[0, 1],
                    traj[0, 2],
                    color=colors[idx],
                    marker="o",
                    s=80,
                    edgecolors="black",
                )
                # Mark Terminal Condition (Cross)
                ax.scatter(
                    traj[-1, 0],
                    traj[-1, 1],
                    traj[-1, 2],
                    color=colors[idx],
                    marker="X",
                    s=80,
                    edgecolors="black",
                )
            else:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=colors[idx],
                    label=name,
                    linewidth=2,
                    alpha=0.8,
                )
                ax.scatter(
                    traj[0, 0],
                    traj[0, 1],
                    color=colors[idx],
                    marker="o",
                    s=80,
                    edgecolors="black",
                )
                ax.scatter(
                    traj[-1, 0],
                    traj[-1, 1],
                    color=colors[idx],
                    marker="X",
                    s=80,
                    edgecolors="black",
                )

            successful_plots += 1

        except Exception as e:
            print(f"Error processing '{file_path}': {e}")

    if successful_plots > 0:
        ax.set_title("Comparative Phase Space Rollout Analysis", fontsize=14)
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")

        # Push the legend outside the plot area so it doesn't occlude trajectory data
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), title="Rollout Files")

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid trajectories were plotted.")


if __name__ == "__main__":
    # Populate this list with the paths to the .npy rollouts you want to compare
    files_to_compare = [
        "wiping_data/wprr_demo1.npy",  # Ground Truth Demonstration
        "ileed_rollout.npy",  # ILEED Output
    ]

    plot_multiple_rollouts(files_to_compare)
