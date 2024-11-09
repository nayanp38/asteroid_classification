import os
import matplotlib.pyplot as plt
import random
import numpy as np

plt.title('PCA Augmentation Until 200 Images')
plt.xlabel('Wavelength')
plt.ylabel('Normalized Reflectance')

plt.xlim(0.4, 2.5)
plt.ylim(0.5, 2)


bd_class = 'L'
class_dir = f'data/cleaned_0.4/{bd_class}'

for graph in os.listdir(class_dir):
    number = graph[:-4]
    spectrum_filepath = f'DeMeo2009data/{number}.txt'
    with open(spectrum_filepath, 'r') as file:
        data = [line.split() for line in file.readlines()]
    wavelength = [float(row[0]) for row in data]
    reflectance = [float(row[1]) for row in data]

    color = plt.cm.viridis(random.random())

    plt.plot(wavelength, reflectance, color=color, alpha=0.5)


for i in range(100-len(os.listdir(class_dir))):
    random_file = random.choice(os.listdir(class_dir))
    number = random_file[:-4]
    spectrum_filepath = f'DeMeo2009data/{number}.txt'
    with open(spectrum_filepath, 'r') as file:
        data = [line.split() for line in file.readlines()]
    wavelength = [float(row[0]) for row in data]
    reflectance = [float(row[1]) for row in data]

    mu = 0
    sigma = 0.015
    augmented_y = [y + np.random.normal(mu, sigma) for y in reflectance]

    color = plt.cm.viridis(random.random())

    plt.plot(wavelength, augmented_y, color=color, alpha=0.5)


for i in range(100-len(os.listdir(class_dir))):
    random_file = random.choice(os.listdir(class_dir))
    number = random_file[:-4]
    spectrum_filepath = f'DeMeo2009data/{number}.txt'
    with open(spectrum_filepath, 'r') as file:
        data = [line.split() for line in file.readlines()]
    wavelength = [float(row[0]) for row in data]
    reflectance = [float(row[1]) for row in data]

    shift = random.uniform(-0.05, 0.05)

    augmented_y = [y + shift for y in reflectance]

    color = plt.cm.viridis(random.random())

    plt.plot(wavelength, augmented_y, color=color, alpha=0.5)


# plt.savefig('augment_comparison_imgs/L_200_pca.png')
plt.show()