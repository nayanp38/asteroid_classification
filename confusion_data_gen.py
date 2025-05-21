import random
from sklearn import metrics
import os
from torchvision import transforms
import numpy
import matplotlib.pyplot as plt
import random

all_classes = ['S', 'Sa', 'Sq', 'Sr', 'Sv',
               'B', 'C', 'Cb', 'Cg', 'Cgh', 'Ch',
               'X', 'Xc', 'Xe', 'Xk',
               'D', 'K', 'L', 'T', 'A', 'O', 'Q', 'R', 'V', 'Xn']

all_classes = sorted(all_classes)
print(all_classes)

def generate_predictions_and_ground_truth(data):
    """
    Generates two lists: predictions and ground truths based on input class data.

    Parameters:
        data (list of dicts): Each dict should have 'Class', '# Correct', and '# Occurrences'

    Returns:
        tuple: (predictions list, ground truth list)
    """
    predictions = []
    ground_truth = []

    classes = [entry['Class'] for entry in data]

    for entry in data:
        true_class = entry['Class']
        correct = int(entry['# Correct'])
        total = int(entry['# Occurrences'])
        incorrect = total - correct

        # Add correct predictions
        predictions.extend([true_class] * correct)
        ground_truth.extend([true_class] * correct)

        # Add incorrect predictions
        for _ in range(incorrect):
            wrong_classes = [c for c in classes if c != true_class]
            predicted_class = random.choice(wrong_classes)
            predictions.append(predicted_class)
            ground_truth.append(true_class)

    return predictions, ground_truth

# Example data
data = [
    {"Class": "A", "# Correct": 1, "# Occurrences": 1},
    {"Class": "B", "# Correct": 6, "# Occurrences": 7},
    {"Class": "C", "# Correct": 10, "# Occurrences": 14},
    {"Class": "Cb", "# Correct": 1, "# Occurrences": 1},
    {"Class": "Cg", "# Correct": 0, "# Occurrences": 1},
    {"Class": "Cgh", "# Correct": 0, "# Occurrences": 0},
    {"Class": "Ch", "# Correct": 2, "# Occurrences": 2},
    {"Class": "D", "# Correct": 3, "# Occurrences": 6},
    {"Class": "K", "# Correct": 0, "# Occurrences": 0},
    {"Class": "L", "# Correct": 10, "# Occurrences": 12},
    {"Class": "Q", "# Correct": 25, "# Occurrences": 36},
    {"Class": "R", "# Correct": 0, "# Occurrences": 1},
    {"Class": "S", "# Correct": 64, "# Occurrences": 68},
    {"Class": "Sa", "# Correct": 1, "# Occurrences": 1},
    {"Class": "Sq", "# Correct": 23, "# Occurrences": 31},
    {"Class": "Sr", "# Correct": 21, "# Occurrences": 23},
    {"Class": "Sv", "# Correct": 2, "# Occurrences": 3},
    {"Class": "U", "# Correct": 0, "# Occurrences": 4},
    {"Class": "V", "# Correct": 11, "# Occurrences": 12},
    {"Class": "X", "# Correct": 9, "# Occurrences": 15},
    {"Class": "Xc", "# Correct": 1, "# Occurrences": 1},
    {"Class": "Xe", "# Correct": 2, "# Occurrences": 4},
    {"Class": "Xk", "# Correct": 0, "# Occurrences": 4},
    {"Class": "Xn", "# Correct": 1, "# Occurrences": 2},
]

# Generate predictions and ground truths
predictions, ground_truth = generate_predictions_and_ground_truth(data)

print("Predictions List:", predictions)
print("Ground Truth List:", ground_truth)

all_pred = predictions
all_true = ground_truth

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
# plt.savefig(f'images/MITHNEOS_dataset_1_conf_{accuracy:.2%}.png', bbox_inches='tight')
plt.show()
'''
with open('data/updated_pred_true', 'w') as file:
    # Write the header
    file.write("Predicted\tGround Truth\n")

    # Write each pair of predicted and ground truth values
    for pred, truth in zip(all_pred, all_true):
        file.write(f"{pred}\t{truth}\n")
'''