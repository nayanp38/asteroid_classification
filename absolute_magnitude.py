import os
import matplotlib.pyplot as plt
from graph_generator import get_abs_mag, extract_number_from_filename, get_filename_from_number, get_diameter
import numpy as np


# Placeholder functions, replace with your actual implementations

def process_directory(directory):
    abs_mags = []
    diameters = []
    directory_labels = []

    for filename in os.listdir(directory):
        if "augmented" not in filename:
            number = extract_number_from_filename(get_filename_from_number(filename))
            abs_mag = get_abs_mag(number)
            diameter = get_diameter(number)

            if abs_mag is not None and diameter is not None:
                abs_mags.append(abs_mag)
                diameters.append(diameter)
                directory_labels.append(directory)

    return abs_mags, diameters, directory_labels


def main():
    main_directory = "graphs"
    sub_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    for sub_directory in sub_directories:
        directory_path = os.path.join(main_directory, sub_directory)
        abs_mags, diameters, directory_labels = process_directory(directory_path)

        if abs_mags and diameters:
            # Plotting as a scatter plot with color-coded points
            scatter_plot = plt.scatter(diameters, abs_mags, c=np.arange(len(diameters)), label=sub_directory)

    # Customize plot
    plt.xlabel("Diameter")
    plt.ylabel("Absolute Magnitude")
    plt.title("Scatter Plot of Diameter vs Absolute Magnitude")
    plt.legend()

    # Add colorbar for better visualization of directories
    plt.colorbar(scatter_plot, label='Directory Index')

    plt.show()


if __name__ == "__main__":
    main()
