import os
import numpy as np
import matplotlib.pyplot as plt
import re

root_path = '../data/mithneos_test'

def create_visnir_graph(file_path, extracted_number, visnir=True):
    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]
    x_values = [float(row[0]) for row in data if len(row) > 1]
    y_values = [float(row[1]) for row in data if len(row) > 1]
    y_avg = np.mean(y_values)
    deviation = 0.4

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
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

    plt.savefig(f'../data/mithneos_graphs/{extracted_number}.png')  # Save the temporary graph for download

    plt.clf()


def extract_id(filename):
    # Define regex patterns for the two formats
    pattern1 = r'a(\d+)\.visnir\.txt'  # Matches format like a{number_id}.visnir.txt
    pattern2 = r'au([a-zA-Z0-9]+)\.visnir\.txt'  # Matches format like au{numberandletter_id}.visnir.txt

    # Try matching the first pattern
    match1 = re.match(pattern1, filename)
    if match1:
        return str(int(match1.group(1)))

    # Try matching the second pattern
    match2 = re.match(pattern2, filename)
    if match2:
        return match2.group(1)

    # If no pattern matches, return None or raise an error
    return None


filenames = os.listdir(root_path)
count = 0
for filename in filenames:
    extracted_id = extract_id(filename)
    print(extracted_id)
    create_visnir_graph(os.path.join(root_path, filename), extracted_id)
    count += 1
    print(count)
