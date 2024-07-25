import os


def delete_files_until_limit(directory_path, limit=200):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Calculate the number of files to delete
    num_files_to_delete = max(0, len(files) - limit)

    # Sort the files by modification time (oldest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)))

    # Delete the oldest files until the limit is reached
    for i in range(num_files_to_delete):
        file_to_delete = files[i]
        file_path = os.path.join(directory_path, file_to_delete)

        try:
            # Remove the file
            os.remove(file_path)
            print(f"Deleted: {file_to_delete}")
        except Exception as e:
            print(f"Error deleting {file_to_delete}: {e}")


# Replace 'path/to/your/directory' with the actual path to your directory
directory_path = 'gaussian_graphs/S'

# Specify the limit (default is 200)
delete_files_until_limit(directory_path, limit=500)