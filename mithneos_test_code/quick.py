import os


def remove_extra_pth(directory):
    # Iterate over all the items in the main directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            # Check if the filename ends with '.pth.pth'
            if filename.endswith('.pth.pth'):
                # Create the new filename by removing the extra '.pth'
                new_filename = filename[:-4]  # Remove the last 4 characters ('.pth')
                new_file_path = os.path.join(directory, new_filename)

                # Rename the file to remove the extra '.pth'
                os.rename(file_path, new_file_path)
                print(f'Renamed: {filename} -> {new_filename}')


# Example usage
main_directory = '../model_dicts'
remove_extra_pth(main_directory)
