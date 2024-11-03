import os
def find_keys_with_v_in_values(input_dict):
    # Create a list to store keys whose values contain 'V'
    keys_with_v = []

    # Iterate through the dictionary
    for key, value in input_dict.items():
        # Check if the letter "V" (case-insensitive) is in the value
        if 'x' in str(value).lower():  # Convert value to string and check for 'v'
            keys_with_v.append(key)

    return keys_with_v


def find_common_keys_in_second_dict(keys_with_v, second_dict):
    # Create a list to store common keys
    common_keys = []

    # Iterate through the list of keys with 'V' and check if they exist in the second dictionary
    for key in keys_with_v:
        if key in second_dict:
            common_keys.append(key)

    return common_keys

true_classifications = {}

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

# Call the function
keys_with_v = find_keys_with_v_in_values(true_classifications)

# Output the keys
print(f"Keys with 'V' in their values: {keys_with_v}")




predicted_classifications = {}

# Read the file and populate the dictionary
with open('classifications.txt', 'r') as file:
    for line in file:
        # Split each line by space and assign key and value
        key, value = line.strip().split(' ', 1)  # Using 1 to ensure only the first space splits the line
        predicted_classifications[key] = value


data_path = os.listdir('../data/cleaned_0.4')
training_nums = []
for folder in data_path:
    training_nums.extend(os.listdir(os.path.join('../data/cleaned_0.4', folder)))

for index, num in enumerate(training_nums):
    training_nums[index] = num[:-4]

duplicates = []
for num in training_nums:
    if num in predicted_classifications:
        duplicates.append(num)
        del predicted_classifications[num]

common_keys = find_common_keys_in_second_dict(keys_with_v, predicted_classifications)

print('common keys:')
print(common_keys)