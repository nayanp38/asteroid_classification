import os
import numpy as np
import matplotlib.pyplot as plt
import re

root_path = '../data/mithneos_test'



filenames = os.listdir(root_path)
'''
for filename in filenames:
    extracted_id = extract_id(filename)
    print(extracted_id)
    create_visnir_graph(os.path.join(root_path, filename), extracted_id)
'''

file_path = '../data/mithneos_test/a000512.visnir.txt'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]
x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]
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
plt.show()

print(x_values)
print(y_values)