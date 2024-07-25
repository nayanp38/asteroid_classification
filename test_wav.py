from model import GraphModel, GraphWavModel
from graph_dataset import GraphDataset, GraphWavDataset
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

graph_directory = f'C:/Users/Nayan_Patel/PycharmProjects/asteroid/visnir_graphs_0.4_from_avg'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a dataset and dataloader for testing
test_dataset = GraphWavDataset(root_dir=graph_directory, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the model and load the trained weights
num_classes = len(os.listdir(graph_directory))
model = GraphWavModel(num_classes)
model.load_state_dict(torch.load(f'model_dicts/visnir_graphs_wavs_0.4_from_avg_epoch20.pth'))
model.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Testing loop
with torch.no_grad():
    for images, wavs, labels in test_dataloader:
        outputs = model(images, wavs)
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
# plt.savefig('confusion_v4_albedos.png')
plt.show()
