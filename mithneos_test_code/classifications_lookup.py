predicted_classifications = {}

# Read the file and populate the dictionary
with open('classifications.txt', 'r') as file:
    for line in file:
        # Split each line by space and assign key and value
        key, value = line.strip().split(' ', 1)  # Using 1 to ensure only the first space splits the line
        predicted_classifications[key] = value

# Print the dictionary to verify
asteroid = '1747'

true_classifications = {}

# Read the file and populate the dictionary
with open('../data/MITHNEOS.txt', 'r') as file:
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

# Print the resulting dictionary
'''
print(true_classifications)
# Print the dictionary to verify
print(predicted_classifications[asteroid])
print(true_classifications[asteroid])
'''

correct_count = 7
total_count = 0

# Dictionaries to store the count of appearances and correct predictions for each ID
id_count = {}
correct_id_count = {}

for asteroid in predicted_classifications:
    corr = False
    pred = predicted_classifications[asteroid]
    true = true_classifications[asteroid]

    if true.startswith('"'):
        # Remove the quote and split the ID by commas
        id_parts = true[1:-1].split(',')
    else:
        id_parts = [true]  # Treat it as a single ID

    # Process each part (ID) separately
    for id_part in id_parts:
        # Remove any 'w' characters from the ID
        processed_id = id_part.replace('w', '')

        # Update the count of how many times this ID has come up
        if processed_id in id_count:
            id_count[processed_id] += 1
        else:
            id_count[processed_id] = 1

        # Compare the processed ID to the predicted ID
        if processed_id[0] == pred[0]:
            correct_count += 1
            corr = True

            # Update the count of how many times this ID was predicted correctly
            if processed_id in correct_id_count:
                correct_id_count[processed_id] += 1
            else:
                correct_id_count[processed_id] = 1

    if corr:
        print(f'{asteroid}: {pred}, {true} | CORRECT')
    else:
        print(f'{asteroid}: {pred}, {true} | WRONG')

    total_count += 1

print(f'{correct_count} / {total_count} = {correct_count / total_count}')

# Print the summary of counts for each ID
for id_key in id_count:
    correct_predictions = correct_id_count.get(id_key, 0)
    total_predictions = id_count[id_key]
    print(
        f'ID {id_key}: {correct_predictions} correct predictions out of {total_predictions} times ({correct_predictions / total_predictions:.2%} accuracy)')
