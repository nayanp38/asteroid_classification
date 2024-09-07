import os
import re
import rocks
from visnir_graph_generator import lookup_type
import matplotlib.pyplot as plt

data_root = 'data/visnir_graphs_0.4_from_avg_albedo'

with open('scraped_albedos.txt', 'r') as file:
    data = [line.split() for line in file.readlines()]

nums = [int(row[0]) for row in data]
albedos = [float(row[1]) for row in data]
types = []
selected_albedos = []

for index, asteroid in enumerate(nums):
    type = lookup_type(asteroid)
    albedo = albedos[index]
    types.append(type)
    selected_albedos.append(albedo)

plt.scatter(types, selected_albedos)
plt.show()