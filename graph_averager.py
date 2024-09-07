import numpy as np
import os
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def fetch_xy_from_file(filepath):
    """
    This function reads a text file and returns x and y values as numpy arrays.
    It assumes the text file has two columns: first for x values and second for y values.
    """
    with open(filepath, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]
    return np.array(x_values), np.array(y_values)


def process_directory(directory_path, type):
    """
    This function processes all text files in a given directory, reads their x and y values,
    and computes the average y value for each x value across all files.
    """
    all_x_values = []
    y_values_list = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        pattern = re.compile(r'(\d+)\.png')
        match = pattern.match(filename)
        number = int(match.group(1))
        data_filename = f'{number}.txt'
        if data_filename.endswith(".txt"):  # Assuming the text files have a .txt extension
            filepath = os.path.join('DeMeo2009data', data_filename)

            # Fetch x and y values from the file
            x_values, y_values = fetch_xy_from_file(filepath)

            all_x_values.extend(x_values)
            y_values_list.append((x_values, y_values))

    if not all_x_values:
        print("No valid data files found.")
        return

    # Get a unique set of x-values for interpolation
    common_x_values = np.unique(np.array(all_x_values))

    # Interpolate y-values to the common set of x-values
    interpolated_y_values_list = []
    for x_values, y_values in y_values_list:
        f = interp1d(x_values, y_values, bounds_error=False, fill_value=np.nan)
        interpolated_y_values = f(common_x_values)
        interpolated_y_values_list.append(interpolated_y_values)

    # Convert list to numpy array for easier manipulation
    interpolated_y_values_array = np.array(interpolated_y_values_list)

    # Compute the average y-values, ignoring NaNs
    average_y_values = np.nanmean(interpolated_y_values_array, axis=0)

    # Print or return the results
    for x, y in zip(common_x_values, average_y_values):
        print(f"x: {x}, avg_y: {y}")

    y_avg = np.mean(average_y_values)
    deviation = 0.4

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    x_min, x_max = 0.4, 2.5
    y_min, y_max = y_avg - deviation, y_avg + deviation
    with open(f'data/averages_raw/{type}.txt', 'w') as file:
        for val1, val2 in zip(common_x_values, average_y_values):
            file.write(f"{val1} {val2}\n")
    '''
    plt.plot(common_x_values, average_y_values)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, 'average.png'))  # Save the temporary graph for download

    plt.clf()
    '''

# Example usage:


directory_path = "data/collapsed_0.4_from_avg"
for dir in os.listdir(directory_path):
    process_directory(os.path.join(directory_path, dir), dir)
