import os

# Specify the path to the encompassing directory
encompassing_path = 'data/demeo_mithneos_20'

# Iterate through sub-directories
def count_files  (encompassing_directory):
    total_classes = len(os.listdir(encompassing_directory))
    total_files = 0
    for sub_directory in os.listdir(encompassing_directory):
        sub_directory_path = os.path.join(encompassing_directory, sub_directory)

        # Check if it's a sub-directory
        if os.path.isdir(sub_directory_path):
            # Count the number of files in the sub-directory
            files_count = len(os.listdir(sub_directory_path))

            # Print the sub-directory name and the number of files
            print(f"{sub_directory}: {files_count} files")
            total_files += files_count
    print(f"Total: {total_files} files")
    print(f'in {total_classes} classes')


def count_files_without_augmented(main_directory):
    # Get a list of all subdirectories in the main directory
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    # Loop through each subdirectory
    total_count = 0
    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)

        files_without_augmented = [f for f in os.listdir(subdir_path) if "augmented" not in f]

        # Display the count for the current subdirectory
        count = len(files_without_augmented)
        print(f"{subdir}: {count}")

        # Update the total count
        total_count += count

    # Display the total count across all subdirectories
    print("\nTotal count across all subdirectories:", total_count)


count_files(encompassing_path)