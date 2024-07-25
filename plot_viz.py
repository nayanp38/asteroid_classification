import matplotlib.pyplot as plt
import numpy as np


file_path = 'smass2/a000246.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='A')

file_path = 'smass2/a000002.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='B')

file_path = 'smass2/a000001.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='C')

file_path = 'smass2/a003885.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='Cg')

file_path = 'smass2/a000013.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='Ch')

file_path = 'smass2/a000199.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='D')

file_path = 'smass2/a000044.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='E')

file_path = 'smass2/a000015.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='K')

file_path = 'smass2/a000172.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='L')

file_path = 'smass2/a000016.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='M')

file_path = 'smass2/a003628.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='O')

file_path = 'smass2/a000046.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='P')

file_path = 'smass2/a005242.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='Q')

file_path = 'smass2/a000349.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='R')

file_path = 'smass2/a003417.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='S')

file_path = 'smass2/a002045.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='V')

file_path = 'smass2/a003533.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='X')

file_path = 'smass2/a000269.[2]'
with open(file_path, 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values, label='Z')


# Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
x_min, x_max = 0.4, 1.0
y_min, y_max = 0.5, 1.5
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Remove grid marks, axes labels, and title
plt.grid(False)
plt.xticks(np.arange(0.4, 1.1, 0.1))  # Remove x-axis labels
plt.yticks(np.arange(0.5, 1.6, 0.1))  # Remove y-axis labels
plt.xlabel('Wavelength µm')
plt.ylabel('Normalized Reflectance')
plt.title('Spectral Data for an Asteroid in Each Class')
plt.legend()
plt.savefig('all_class_data.png')
plt.show()