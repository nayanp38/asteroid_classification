import csv
from sklearn.metrics import classification_report

with open('data/updated_pred_true', 'r') as file:
    # Skip the header
    next(file)

    data = [line.split() for line in file.readlines()]

    all_pred = [(row[0]) for row in data]
    all_true = [(row[1]) for row in data]


# Generate the classification report
report = classification_report(all_true, all_pred, output_dict=True)

# Define the CSV file path
csv_file_path = 'data/classification_metrics.csv'

# Open the CSV file and write the data
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["Class", "Precision", "Recall", "F1 Score"])

    # Write the precision, recall, and F1 score for each class
    for class_label, metrics in report.items():
        if class_label in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        writer.writerow([class_label, metrics['precision'], metrics['recall'], metrics['f1-score']])

print(f"Classification report saved to {csv_file_path}")


