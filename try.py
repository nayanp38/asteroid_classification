import re
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
'''
transform = transforms.Compose([
    transforms.Resize((128, 128)),
])

image = Image.open('gaussian_graphs/C/1.png') # Convert to grayscale

new_image = transform(image)

plt.imshow(new_image)  # Assuming the image is grayscale
plt.axis('off')  # Hide axes
plt.show()

plt.clf()


with open('smass2/a000001.[2]', 'r') as file:
    data = [line.split() for line in file.readlines()]

x_values = [float(row[0]) for row in data]
y_values = [float(row[1]) for row in data]

plt.plot(x_values, y_values)

# Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5

x_min, x_max = 0.4, 1.0
y_min, y_max = 0.5, 1.5

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Remove grid marks, axes labels, and title
plt.grid(False)
plt.xticks([])  # Remove x-axis labels
plt.yticks([])  # Remove y-axis labels
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.gcf().set_size_inches(128 / plt.gcf().dpi, 128 / plt.gcf().dpi)
plt.show()

def find_file_with_longest_column(directory):
    max_length = 0
    max_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the current item is a file
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Read the first column of each file
                column_lengths = [line.split()[0] for line in file.readlines() if line.strip() and float(line.split()[0]) != 0]

                # Find the maximum length
                current_max_length = len(column_lengths)

                # Update max_length and max_file if needed
                if current_max_length > max_length:
                    max_length = current_max_length
                    max_file = filename

    return max_file


# Example usage:
directory = 'smass2'
result = find_file_with_longest_column(directory)
print("File with the longest first column:", result)
'''
print(os.listdir('graphs/A'))