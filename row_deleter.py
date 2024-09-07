import os

def is_negative_one(value):
    """
    This function checks if a given value is equivalent to -1,
    considering various float representations.
    """
    try:
        return float(value) == -1.0
    except ValueError:
        return False

def clean_file(filepath):
    """
    This function reads a text file, removes rows that contain the value -1 (in any float representation),
    and writes the cleaned data back to the file.
    """
    cleaned_data = []
    with open(filepath, 'r') as file:
        for line in file:
            values = line.split()
            if not any(is_negative_one(value) for value in values):
                cleaned_data.append(line)

    # Write the cleaned data back to the file
    with open(filepath, 'w') as file:
        file.writelines(cleaned_data)

def process_directory(directory_path):
    """
    This function processes all text files in a given directory,
    removing rows that contain the value -1 (in any float representation) in each file.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Assuming the text files have a .txt extension
            filepath = os.path.join(directory_path, filename)
            clean_file(filepath)

# Example usage:
directory_path = "DeMeo2009data"
process_directory(directory_path)
