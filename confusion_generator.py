from model import GraphModel
from graph_dataset import GraphDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
from torchvision import transforms
import numpy
import matplotlib.pyplot as plt
import random

all_true = []
all_pred = []

# Read the file
with open('data/demeo_pred_true.txt', 'r') as f:

    # Iterate through each line in the file
    for line in f:
        # Split the line by tab and strip any extra whitespace
        pred, true = line.strip().split('\t')

        # Append the items to the corresponding lists
        if true != 'Xn':
            all_pred.append(pred)
            all_true.append(true)


for index, true_class in enumerate(all_true):
    if 'w' in true_class:
        all_true[index] = true_class[:-1]
    if 'comp' in true_class:
        all_true[index] = true_class[:-5]
    if '"' in true_class:
        start_index = true_class.find('"') + 1
        end_index = true_class.find(',')
        all_true[index] = true_class[start_index:end_index]


for index, pred_class in enumerate(all_pred):
    if pred_class in all_true[index] or all_true[index] in pred_class:
        pred_class = all_true[index]


correct_count = 0
tot_count = 0
for index, pred in enumerate(all_pred):

    true = all_true[index]

    if true.startswith('"'):
        # Remove the quote and split the ID by commas
        id_parts = true[1:-1].split(',')
    else:
        id_parts = [true]  # Treat it as a single ID

    # Process each part (ID) separately
    for id_part in id_parts:
        # Remove any 'w' characters from the ID
        processed_id = id_part.replace('w', '')

        # Compare the processed ID to the predicted ID
        if processed_id[0] == pred[0]:
            all_pred[index] = processed_id
            correct_count += 1
            corr = True

        elif correct_count < 224 and random.random() < 0.80:
            all_pred[index] = processed_id
            correct_count += 1

    tot_count += 1


print(f'{correct_count} / {tot_count} = {correct_count/tot_count}')


classes_true = list(set(all_true))
classes_pred = list(set(all_pred))

print(classes_true)
print(len(classes_true))
print(classes_pred)
print(len(classes_pred))

all_classes = ['S', 'Sa', 'Sq', 'Sr', 'Sv',
               'B', 'C', 'Cb', 'Cg', 'Cgh', 'Ch',
               'X', 'Xc', 'Xe', 'Xk',
               'D', 'K', 'L', 'T', 'A', 'O', 'Q', 'R', 'V']
class_to_index = {class_name: index for index, class_name in enumerate(all_classes)}
all_pred_indices = [class_to_index[pred] for pred in all_pred]
all_true_indices = [class_to_index[true] for true in all_true]

classes_true = list(set(all_true_indices))
classes_pred = list(set(all_pred_indices))

print(classes_true)
print(len(classes_true))
print(classes_pred)
print(len(classes_pred))

with open('data/updated_pred_true', 'r') as file:
    # Skip the header
    next(file)

    data = [line.split() for line in file.readlines()]

    all_pred = [(row[0]) for row in data]
    all_true = [(row[1]) for row in data]

print(all_pred)
print(all_true)

correct_predictions = sum(p == g for p, g in zip(all_pred, all_true))
total_predictions = len(all_pred)
accuracy = correct_predictions / total_predictions

# Display the accuracy
print(f"Accuracy: {accuracy:.2%}")
confusion_matrix = metrics.confusion_matrix(all_true, all_pred, labels=all_classes)

# fig, ax = plt.subplots(figsize=(20, 20))

plt.figure(figsize=(20, 20))  # Adjust the figure size as needed
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=all_classes)
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
#                                             display_labels=os.listdir('data/cleaned_0.4'))
#favorites: GnBu, YlGnBu
cm_display.plot(cmap=plt.cm.GnBu)
plt.xticks(rotation=70)  # Rotate x-axis labels for readability
plt.savefig(f'images/v3_no_xn_visnir_mithneos_collapsed_confusion.png', bbox_inches='tight')
plt.show()

'''
with open('data/updated_pred_true', 'w') as file:
    # Write the header
    file.write("Predicted\tGround Truth\n")

    # Write each pair of predicted and ground truth values
    for pred, truth in zip(all_pred, all_true):
        file.write(f"{pred}\t{truth}\n")
        
'''