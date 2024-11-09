import os

from matplotlib import pyplot as plt

plt.title('1685 (Sq Type)')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Normalized Reflectance')

plt.xlim(0.4, 2.5)
plt.ylim(0.5, 1.5)

spectrum_filepath = f'data/mithneos_test/a001685.visnir.txt'
with open(spectrum_filepath, 'r') as file:
    data = [line.split() for line in file.readlines()]
wavelength = [float(row[0]) for row in data]
reflectance = [float(row[1]) for row in data]

plt.plot(wavelength, reflectance)

plt.savefig('images/1685_Sq_graph.png')

plt.show()
