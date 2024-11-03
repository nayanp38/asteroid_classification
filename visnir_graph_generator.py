import os
import matplotlib.pyplot as plt
import rocks
import random
import re
import numpy as np


# Function to create a graph from a text file
def create_graph(file_path, extracted_number):
    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    x_min, x_max = 0.4, 1.0
    y_min, y_max = 0.5, 1.5

    plt.plot(x_values, y_values)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    plt.savefig(f'{extracted_number}.png')  # Save the temporary graph for download

    plt.clf()


def create_visnir_graph(file_path, extracted_number, visnir=True):
    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]
    y_avg = np.mean(y_values)
    deviation = 0.4

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    if not visnir:
        x_min, x_max = 0.4, 1.0
        y_min, y_max = 0.5, 1.5
    else:
        x_min, x_max = 0.4, 2.5
        y_min, y_max = y_avg - deviation, y_avg + deviation

    plt.plot(x_values, y_values)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    plt.savefig(f'{extracted_number}.png')  # Save the temporary graph for download

    plt.clf()


def augment_graph(label_root, type):
    file_path = os.path.join('data/averages_raw', f'{type}.txt')

    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]
    minim = min(y_values)
    maxim = max(y_values)

    y_avg = np.mean(y_values)
    deviation = 0.4

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5

    mu = 0
    sigma = 0.015
    augmented_y = [y + np.random.normal(mu, sigma) for y in y_values]

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    x_min, x_max = 0.4, 2.5
    y_min, y_max = y_avg - deviation, y_avg + deviation

    plt.plot(x_values, augmented_y)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    # Save the augmented graph with a unique filename

    output_directory = os.path.join('data/augmented_0.4', type)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    augmented_filename = f'{type}_augmented_{len(os.listdir(output_directory))}.png'
    augmented_filepath = os.path.join(output_directory, augmented_filename)
    plt.savefig(augmented_filepath)
    print(f'Created: {augmented_filepath}')

    plt.clf()


def png_to_number(filename):
    match = re.search(r'\d+', filename)

    if match:
        # Extracted number is in the first capturing group
        extracted_number = match.group(0)
        print(filename)
        print(extracted_number)
        return int(extracted_number)
    else:
        # Return None if no number is found
        print('no number found')
        return None


# Function to extract a number from a filename

def extract_number_from_filename(file_name, num_unnumbered=0):
    pattern = re.compile(r'(\d+)\.txt')
    match = pattern.match(file_name)
    if match:
        return int(match.group(1))
    else:
        num_unnumbered += 1
        return 'unnumbered' + str(num_unnumbered)


def get_filename_from_number(filename, no_augments=False):
    if filename.endswith('.png'):
        # Step 1: Extract the number from the original filename
        match = re.search(r'\d+', filename)
        number = int(match.group(0))
        # Step 2: Format the new filename with the specified pattern
        formatted_number = f'{number}.txt'
        return formatted_number


# Function to get the type based on the extracted number
def getType(number):
    try:
        tax = rocks.Rock(number).taxonomy.class_.value if rocks.Rock(number).taxonomy.class_.value else None
        return tax
    except Exception as e:
        print(f"Error in getType function: {e}")
        return None


def get_abs_mag(img):
    number = png_to_number(img)
    try:
        abs_mag = rocks.Rock(number).absolute_magnitude.value
        return abs_mag
    except Exception as e:
        print(f'Error in get_abs_mag function: {e}')
        return None


def get_diameter(img):
    number = png_to_number(img)
    try:
        diameter = rocks.Rock(number).diameter.value
        return diameter
    except Exception as e:
        print(f'Error in diameter function: {e}')
        return None


def get_albedo(img):
    number = png_to_number(img)
    try:
        albedo = rocks.Rock(number).albedo.value
        return albedo
    except Exception as e:
        print(f'Error in albedo function: {e}')
        return None


# Function to store the graph in a directory based on the type
def store_graph(graph_path, graph_type, destination_directory):
    type_directory = os.path.join(destination_directory, graph_type)
    os.makedirs(type_directory, exist_ok=True)

    destination_path = os.path.join(type_directory, os.path.basename(graph_path))
    os.replace(graph_path, destination_path)


def lookup_type(number):
    with open('demeotax.tab', 'r') as file:
        data = [line.split() for line in file.readlines()]
    asteroid_nums = [float(row[0]) for row in data]
    types = [row[3] for row in data]
    index = asteroid_nums.index(number)
    return types[index]

# Specify the path to the directory containing text files
input_dir = 'DeMeo2009data'

# Specify the path to the directory where graphs will be stored
output_dir = f"data/cleaned_0.4"


def generate_data(input_directory, output_directory):
    count = 0
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)
        print(file_name)

        # Task 2: Extract a number from the filename
        extracted_number = extract_number_from_filename(file_name)
        print(extracted_number)

        # Task 3: Get the type based on the extracted number
        graph_type = lookup_type(extracted_number)

        # Task 4: Store the graph in a directory based on the type
        if graph_type:
            # Task 1: Create a graph
            print(extracted_number)
            create_visnir_graph(file_path, extracted_number)

            store_graph(f'{extracted_number}.png', graph_type, output_directory)
            count += 1

        if (count % 100) == 0:
            print(f'Finished {count} items')


# 1263 Files before augmentations
def create_augmentations(directory, target_count):
    for label in os.listdir(directory):
        while len(os.listdir(os.path.join(directory, label))) < target_count:
            augment_graph(os.path.join(directory, label), label)


def delete_augmented_files(main_directory):
    # Iterate through subdirectories in the main directory
    for root, _, files in os.walk(main_directory):
        for file in files:
            # Check if the file name contains "augmented"
            if "augmented" in file:
                if 'pca' not in file:
                    # Construct the full path to the file
                    file_path = os.path.join(root, file)

                    # Delete the file
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")


def rename_files_in_directory(directory):
    # Define the pattern to match the filenames
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern
        # Extract the number from the filename
        number = extract_number_from_filename(filename)
        # Define the new filename
        new_filename = f"{number}.txt"
        # Define the full paths for old and new filenames
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")


if __name__ == "__main__":
    generate_data('smass2', 'data/NOT_VISNIR_test_set')