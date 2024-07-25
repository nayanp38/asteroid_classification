import os
import tabulate

import os
from tabulate import tabulate


def count_files_without_augmented(main_directory):
    # Get a list of all subdirectories in the main directory
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    # Initialize a list to store table rows
    table_data = []

    # Loop through each subdirectory
    total_count = 0
    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)

        # Get a list of all files in the subdirectory without the word "augmented" in the name
        files_without_augmented = [f for f in os.listdir(subdir_path) if "augmented" not in f]

        # Display the count for the current subdirectory
        count = len(files_without_augmented)
        table_data.append([subdir, count])

        # Update the total count
        total_count += count

    # Add a row for the total count
    table_data.append(["Total", total_count])

    # Display the table
    print(tabulate(table_data, headers=["B-DM Class", "Graph Count"], tablefmt="grid"))


# Replace 'path/to/your/main/directory' with the actual path to your main directory
main_directory_path = 'graphs'
count_files_without_augmented(main_directory_path)
