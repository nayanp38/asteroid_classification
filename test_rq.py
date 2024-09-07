import numpy as np

file_path = 'DeMeo2009data/1459.txt'

with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]
y_avg = np.mean(y_values)

print(y_avg)