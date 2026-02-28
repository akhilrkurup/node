import json
import numpy as np
import os


def convert_json_to_npy(json_files):
    """
    Parses a list of JSON files containing dictionaries with 'ee_trans'
    and saves them as .npy files with shape (points, 3).
    """
    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract ee_trans from each dictionary in the list
            # Using a list comprehension to build the (N, 3) matrix
            ee_translations = [item["ee_trans"] for item in data if "ee_trans" in item]

            # Convert to numpy array
            traj_array = np.array(ee_translations)

            # Define output path (changing .json or .txt to .npy)
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}.npy"

            # Save the file
            np.save(output_path, traj_array)
            print(
                f"Successfully converted {file_path} -> {output_path} (Shape: {traj_array.shape})"
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # Add your list of files here
    files_to_convert = [
        "./all_data/stirring_data/demo_new_1.json",
        "./all_data/stirring_data/demo_new_2.json",
        "./all_data/stirring_data/demo_new_3.json",
        "./all_data/stirring_data/demo_new_4.json",
        "./all_data/stirring_data/demo_new_5.json",
    ]

    convert_json_to_npy(files_to_convert)
