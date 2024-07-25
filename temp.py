import matplotlib.pyplot as plt
import os

file_path = 'DeMeo2009data/1.txt'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

# Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
x_min, x_max = 0.4, 1
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

plt.show()