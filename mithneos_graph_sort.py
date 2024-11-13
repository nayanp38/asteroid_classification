import os
import shutil

mithneos_path = 'data/mithneos_graphs'
demeo_path = 'data/demeo_mithneos'

true_classifications = {}

with open('data/MITHNEOS.txt', 'r') as file:
    text = file.read()
for line in text.strip().split('\n'):
    # Split the line by commas and strip whitespace
    parts = [part.strip() for part in line.split('\t')]

    # Extract keys and value
    key1, key2, value = parts

    # If key1 is not blank, map both key1 and key2 to the value
    if key1:
        true_classifications[key1.replace(' ', '')] = value

    # Always map key2 to the value
    true_classifications[key2.replace(' ', '')] = value


for graph in os.listdir(mithneos_path):
    number = graph[:-4]
    asteroid_class = true_classifications[number]

    if asteroid_class.startswith('"'):
        # Find the position of the first double quote and the first comma
        first_quote_index = asteroid_class.find('"')
        first_comma_index = asteroid_class.find(',')

        # If both characters are found, slice the string in between
        if first_quote_index != -1 and first_comma_index != -1:
            asteroid_class = asteroid_class[first_quote_index + 1:first_comma_index]
    # Define the source file and destination directory
    source_file = os.path.join(mithneos_path, graph)
    destination_directory = os.path.join(demeo_path, asteroid_class)
    destination_file = os.path.join(destination_directory, graph)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Copy the file
    shutil.copy2(source_file, destination_file)
