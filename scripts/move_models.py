import os
import shutil

def copy_file_to_directory_with_rename(source_file, destination_dir):
    """Copies a file to a destination directory with a renamed filename.

    The new filename format is <model_name>_<train_iteration>.pt

    Args:
        source_file: The path to the file to be copied.
        destination_dir: The path to the destination directory.
    """
    try:
        # Extract parts from the source path
        parts = source_file.split(os.sep)
        # e.g., ['runs', 'yolo12x', 'train2', 'weights', 'last.pt']
        if len(parts) < 4:
            raise ValueError("Unexpected source_file path format.")

        model_name = parts[1]
        train_iteration = parts[2]
        new_filename = f"{model_name}_{train_iteration}.pt"

        destination_path = os.path.join(destination_dir, new_filename)

        shutil.copy(source_file, destination_path)
        print(f"Successfully copied '{source_file}' to '{destination_path}'")
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
    except PermissionError:
        print(f"Error: Permission denied to copy '{source_file}' to '{destination_dir}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Copy and rename all 5 model weights
copy_file_to_directory_with_rename('runs/yolo12-resnext101-timm/train/weights/last.pt', '../models')
copy_file_to_directory_with_rename('runs/yolo12n/train/weights/last.pt', '../models')
copy_file_to_directory_with_rename('runs/yolo12s/train/weights/last.pt', '../models')
copy_file_to_directory_with_rename('runs/yolo12x/train/weights/last.pt', '../models')
copy_file_to_directory_with_rename('runs/yolo12x/train2/weights/last.pt', '../models')
