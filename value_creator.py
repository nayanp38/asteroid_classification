import random

output_file_path = '500_albedos_created'

# Read diameters from the file
with open('500_albedos_zeroed', 'r') as file:
    diameters = [float(line.strip()) for line in file]

# Group diameters into 18 groups of 200
grouped_diameters = [diameters[i:i+500] for i in range(0, len(diameters), 500)]

# Calculate average diameter for each group excluding zeros
average_diameters = []
for group in grouped_diameters:
    non_zero_group = [d for d in group if d != 0]
    average = sum(non_zero_group) / len(non_zero_group) if non_zero_group else 0
    average_diameters.append(average)

# Replace each 0 with a random variation from -10 to 10

variation = 0.005
for i in range(len(diameters)):
    if diameters[i] == 0:
        group_index = i // 500
        varied_average = average_diameters[group_index] + random.uniform(-variation, variation)
        diameters[i] = abs(varied_average)

with open(output_file_path, 'w') as output_file:
    for diameter in diameters:
        output_file.write(f"{diameter}\n")

# Print the results

print("Average Diameters for Each Group (excluding 0s):", average_diameters)
# print("Diameters after replacing 0s with random variations:", diameters)
