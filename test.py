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

deviation = 0.4
epoch = 20

graph_path = 'data/NOT_VISNIR_mithneos_graphs'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a dataset and dataloader for testing
test_dataset = GraphDataset(root_dir=graph_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the model and load the trained weights
# num_classes = len(os.listdir(graph_directory))
num_classes = 24
model = GraphModel(num_classes)
model.load_state_dict(torch.load(f'model_dicts/final_model/full_model_13'))
model.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Testing loop
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate accuracy
print(predicted_labels)
print(true_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")


confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=test_dataset.get_classes())

cm_display.plot()
# plt.savefig(f'visnir_collapsed_confusion_{accuracy}.png')
plt.show()
