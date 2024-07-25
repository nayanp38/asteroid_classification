import os
import matplotlib.pyplot as plt
import random
import numpy as np
from graph_generator import getType, get_filename_from_number, extract_number_from_filename

label_root = 'gaussian_graphs/V'


def display_one_augment(label_root):
    augmented_files = [filename for filename in os.listdir(label_root) if "augmented" not in filename]
    random_file = random.choice(augmented_files)
    file_name = get_filename_from_number(random_file)
    file_path = os.path.join('smass2', file_name)

    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]

    mu = 0
    sigma = 0.015
    augmented_y = [y + np.random.normal(mu, sigma) for y in y_values]

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    x_min, x_max = 0.4, 1.0
    y_min, y_max = 0.5, 1.5

    # Add transparency and random colors
    color = plt.cm.viridis(random.random())
    plt.plot(x_values, augmented_y, alpha=0.5, color=color, linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')


def display_one_graph(label_root):
    files = [filename for filename in os.listdir(label_root) if "augmented" not in filename]
    random_file = random.choice(files)
    file_name = get_filename_from_number(random_file)
    file_path = os.path.join('smass2', file_name)

    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]


    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    x_min, x_max = 0.4, 1.0
    y_min, y_max = 0.5, 1.5

    # Add transparency and random colors
    color = plt.cm.viridis(random.random())
    plt.plot(x_values, y_values, alpha=0.5, color=color, linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')


if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    for i in range(42):
        display_one_graph(label_root)

    for i in range(158):
        display_one_augment(label_root)

    # plt.savefig('bigger_v_200_gaussian.png')
    plt.show()
