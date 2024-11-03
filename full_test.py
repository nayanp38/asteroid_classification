from model import FullModel
from graph_dataset import FullDataset, Original200Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
from torchvision import transforms
import numpy
import matplotlib.pyplot as plt

graph_path = 'data/NOT_VISNIR_mithneos_graphs'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a dataset and dataloader for testing
test_dataset = Original200Dataset(root_dir=graph_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the model and load the trained weights
num_classes = len(os.listdir(graph_path))
print(num_classes)
model = FullModel(num_classes)
model.load_state_dict(torch.load('model_dicts/500_model/full_model_13'))
model.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []
imgs = []

# Testing loop
with torch.no_grad():
    for images, diameter, abs_mag, albedo, labels in test_dataloader:
        outputs = model(images, diameter, abs_mag, albedo)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        # imgs.extend(imgs.cpu().numpy())

# Calculate accuracy
print(predicted_labels)
print(true_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")


confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=test_dataset.get_classes())

cm_display.plot()
# plt.savefig('500_full_confusion.png')
plt.show()
