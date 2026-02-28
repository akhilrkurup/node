import subprocess
import sys
import os


def run_all_trainings():
    # Define the tasks and data types you want to iterate through
    tasks = ["stirring", "pick"]
    data_types = ["raw", "processed"]

    # Base script to execute
    script_name = "./IROS/code/node_train.py"

    for task in tasks:
        for data in data_types:
            folder_path = f"./IROS/{task}/{data}"

            # Optional: Check if the directory actually exists before trying to run
            if not os.path.isdir(folder_path):
                print(f"⚠️  Skipping {folder_path}: Directory does not exist.\n")
                continue

            print(f"==================================================")
            print(f"🚀 Starting NODE training for: {folder_path}")
            print(f"==================================================")

            try:
                # Run the training script.
                # sys.executable ensures it uses your current Conda environment (glide)
                # check=True ensures that if node_train.py crashes, it raises a CalledProcessError
                subprocess.run(
                    [sys.executable, script_name, "--master_folder", folder_path],
                    check=True,
                )
                print(f"✅ Successfully finished training for {folder_path}\n")

            except subprocess.CalledProcessError as e:
                # If the script throws ANY error, it gets caught here instead of killing the loop
                print(
                    f"❌ ERROR: Training crashed for {folder_path} (Exit code {e.returncode})."
                )
                print("⏭️  Skipping to the next task...\n")

            except KeyboardInterrupt:
                # Allows you to press Ctrl+C to safely abort the entire batch process
                print("\n🛑 Batch execution aborted by user.")
                sys.exit(0)


if __name__ == "__main__":
    print("Starting Automated Batch Training...")
    run_all_trainings()
    print("🎉 All tasks completed!")
