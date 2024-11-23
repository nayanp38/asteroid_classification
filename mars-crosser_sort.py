import re
import os
import csv
import shutil

infile_path = 'data/filtered_mars-crossers.txt'

'''
c
# Open the input file in read mode and the output file in write mode
with open("data/mars-crossers.txt", "r") as infile, open("data/filtered_mars-crossers.txt", "w") as outfile:
    for line in infile:
        # Split the line into words and filter based on the criteria
        filtered_words = [
            word for word in line.split()
            if len(word) <= 2 or (len(word) > 2 and re.search(r'\d', word))
        ]
        # Write the filtered line to the output file
        outfile.write(" ".join(filtered_words) + "\n")


# Open the input file in read mode and the output file in write mode
with open(infile_path, "r") as infile, open("data/formatted_mars-crossers.txt", "w") as outfile:
    for line in infile:
        # Split the line into words and filter based on the criteria
        filtered_words = [word for word in line.split()]

        # Initialize an empty list to store the processed words for the current line
        processed_line = []

        # Iterate through the filtered words
        for word in filtered_words:
            # If the word contains any letter, combine it with the previous word
            if re.search(r'[A-Za-z]', word) and processed_line:
                processed_line[-1] += word  # Combine with the last word in processed_line
            else:
                processed_line.append(word)  # Otherwise, add it as a new word

        # Write the processed line to the output file
        outfile.write(" ".join(processed_line) + "\n")
'''

# Set the path to the directory with filenames to check against
mars_crossers = "data/formatted_mars-crossers.txt"

match_count = {}
total_matches = []

# Get a list of all filenames in the specified directory
for class_dir in os.listdir('data/demeo_mithneos'):
    matching_filenames = []
    filenames_in_directory = os.listdir(os.path.join('data/demeo_mithneos', class_dir))
    filenames_in_directory = [element[:-4] for element in filenames_in_directory]

    with open(mars_crossers, "r") as infile:
        for line in infile:
            # Split the line into words
            words = line.split()
            for word in words:
                # Remove parentheses from the word if they exist
                cleaned_word = re.sub(r'[()]', '', word)
                # Check if the cleaned word matches any filename in the directory
                if cleaned_word in filenames_in_directory:
                    matching_filenames.append(cleaned_word)
    match_count[class_dir] = len(matching_filenames)
    total_matches += matching_filenames


for key in match_count:
    print(f'{key}: {match_count[key]}')

print(total_matches)
print(len(total_matches))



'''
output_file = 'mars_crossers_by_class.csv'

# Write the dictionary to a CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Class', 'Number of Mars-Crossers'])
    # Write each key-value pair
    for key, value in match_count.items():
        writer.writerow([key, value])
        
'''

# Define the source base directory (where files are currently located)
source_base_dir = "data/no_marscrossers_demeo_mithneos"

# Define the target base directory (where files should be moved)
target_base_dir = "data/mars_crossers"

# Loop through each file in the list of matching filenames
for filename in total_matches:
    graph_filename = filename + '.png'
    # Search for the file in the source directory and locate its immediate parent directory
    for root, dirs, files in os.walk(source_base_dir):
        if (graph_filename) in files:
            # Get the immediate parent directory name
            parent_dir = os.path.basename(root)

            # Construct the source and target paths
            source_path = os.path.join(root, graph_filename)
            target_path = os.path.join(target_base_dir, parent_dir, graph_filename)

            # Create the target directory if it doesn't exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Move the file
            shutil.move(source_path, target_path)
            print(f"Moved {graph_filename} to {target_path}")
            break  # Stop searching once the file is found and moved
