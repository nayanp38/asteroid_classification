import random

# Read diameters from the file
input_file_path = '500_albedos_created'
output_file_path = '500_albedos_randomized'

with open(input_file_path, 'r') as file:
    diameters = [float(line.strip()) for line in file]

# Identify chunks of repeated diameters
chunked_diameters = []
current_chunk = []
prev_diameter = None

for diameter in diameters:
    if diameter == prev_diameter:
        current_chunk.append(diameter)
    else:
        if current_chunk:
            chunked_diameters.append(current_chunk)
        current_chunk = [diameter]
    prev_diameter = diameter

if current_chunk:
    chunked_diameters.append(current_chunk)

# Vary the following diameters for each chunk by +-1
for chunk in chunked_diameters:
    average_chunk = sum(chunk) / len(chunk)
    for i in range(1, len(chunk)):
        chunk[i] = average_chunk + random.uniform(-0.001, 0.001)

# Flatten the chunked diameters back to a single list
diameters = [diameter for chunk in chunked_diameters for diameter in chunk]

# Write the updated diameters to the output file
with open(output_file_path, 'w') as output_file:
    for diameter in diameters:
        output_file.write(f"{diameter}\n")

print("Updated diameters written to:", output_file_path)
