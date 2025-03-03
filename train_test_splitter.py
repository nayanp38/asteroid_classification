import os
import shutil
import random
import math


def move_files(source_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over each sub-directory in the main directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Get list of files in the sub-directory
            files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            num_files = len(files)

            # Calculate number of files to move
            num_to_move = round(num_files * 0.20)
            if num_to_move == 0:
                continue  # Skip sub-directories with fewer than 5 files

            # Randomly select files to move
            files_to_move = random.sample(files, num_to_move)

            # Create the corresponding sub-directory in the destination directory
            dest_subdir_path = os.path.join(dest_dir, subdir)
            if not os.path.exists(dest_subdir_path):
                os.makedirs(dest_subdir_path)

            # Move the selected files
            for file_name in files_to_move:
                src_file = os.path.join(subdir_path, file_name)
                dest_file = os.path.join(dest_subdir_path, file_name)
                shutil.move(src_file, dest_file)

            print(f"Moved {num_to_move} files from {subdir_path} to {dest_subdir_path}")


# Define the source directory and destination directory
source_path = 'data/demeo_mithneos_80'
dest_path = 'data/demeo_mithneos_20'

# Call the function
for dir in os.listdir(source_path):
    if not os.path.isdir(os.path.join(dest_path, dir)):
        os.makedirs(os.path.join(dest_path, dir))
move_files(source_path, dest_path)
