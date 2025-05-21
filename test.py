from model import GraphModel
from graph_dataset import TestDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import csv

deviation = 0.4
epoch = 20

graph_path = 'data/mars_crossers'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a dataset and dataloader for testing
test_dataset = TestDataset(root_dir=graph_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the model and load the trained weights
# num_classes = len(os.listdir(graph_directory))
num_classes = 25


# mars crosser dict: 'model_dicts/demeo_mithneos_80/v2_5'
# 80/20 dict: 'model_dicts/no_mars_crossers/v3_8'
model = GraphModel(num_classes)
model.load_state_dict(torch.load(f'model_dicts/demeo_mithneos_80/v2_5'))
model.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []
all_numbers = []

# Testing loop
with torch.no_grad():
    for images, labels, numbers in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_numbers.extend(numbers)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate accuracy
print(predicted_labels)
print(true_labels)
print(all_numbers)
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")

for i, (x, y) in enumerate(zip(predicted_labels, true_labels)):
    if x != y:
        print(all_numbers[i])

'''
for i, (x, y) in enumerate(zip(predicted_labels, true_labels)):
    if x != y:
        print(f"Index {i}: "
              f"Number = {all_numbers[i]}, "
              f"Predicted = {test_dataset.get_classes()[x]}, "
              f"True = {test_dataset.get_classes()[y]}")


predicted Sr, true S:
18882
3255
3430
4995
68350

Predicted Sq, true S:
236716

Predicted S, true Sr
3858

'''

true_labels = [test_dataset.get_classes()[f] for f in true_labels]
predicted_labels = [test_dataset.get_classes()[f] for f in predicted_labels]

rows = zip(all_numbers, true_labels, predicted_labels)
'''
# Write to a CSV file
with open(f'classifications_mithneos_{str(accuracy*100)[:4]}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header (optional)
    writer.writerow(['Number', 'True', 'Predicted'])
    # Write the rows
    writer.writerows(rows)
'''

confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=test_dataset.get_classes())

plt.figure(figsize=(20, 20))  # Adjust the figure size as needed
cm_display.plot(cmap=plt.cm.GnBu)
plt.xticks(rotation=70)  # Rotate x-axis labels for readability
# plt.savefig(f'images/mars_crossers_{str(accuracy*100)[:4]}.png', bbox_inches='tight')
plt.show()
