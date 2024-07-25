import os
import shutil


def move_files_to_main_directory(main_directory, target_files):
    # List to keep track of found files
    found_files = []

    # Walk through all subdirectories in the main directory
    for root, dirs, files in os.walk(main_directory):
        # Skip the main directory itself
        if root == main_directory:
            continue

        for file in files:
            if file in target_files:
                # Construct full file path
                file_path = os.path.join(root, file)
                # Construct new file path in the main directory
                new_file_path = os.path.join(main_directory, file)

                # Move file to the main directory
                shutil.move(file_path, new_file_path)
                print(f"Moved: {file_path} to {new_file_path}")

                # Add the file to the found_files list
                found_files.append(file)

                # Break if we have found all target files
                if len(found_files) == len(target_files):
                    break
        # Break the outer loop if we have found all target files
        if len(found_files) == len(target_files):
            break

    # Report any files not found
    not_found_files = set(target_files) - set(found_files)
    if not_found_files:
        print(f"Files not found: {not_found_files}")
    else:
        print("All target files have been moved.")


# Example usage
main_directory = 'C:/Users/Nayan_Patel/PycharmProjects/asteroid/visnir_graphs_0.4_from_avg_albedo'
target_files = ['512.png', '402.png', '513.png', '532.png',
                '570.png', '606.png', '625.png', '631.png', '653.png',
                '661.png', '706.png', '716.png', '719.png', '720.png', '808.png', '999995.png',
                '999996.png', '999997.png', '999998.png', '999999.png']

move_files_to_main_directory(main_directory, target_files)
