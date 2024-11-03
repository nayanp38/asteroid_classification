import os

predicted_classifications = {}

# Read the file and populate the dictionary
with open('classifications.txt', 'r') as file:
    for line in file:
        # Split each line by space and assign key and value
        key, value = line.strip().split(' ', 1)  # Using 1 to ensure only the first space splits the line
        predicted_classifications[key] = value

# Print the dictionary to verify

print(len(predicted_classifications))

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

print(duplicates)
print(len(predicted_classifications))