from model import GraphModel
from graph_dataset import GraphDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

deviation = 0.4
epoch = 20

graph_path = 'data/demeo_mithneos_20'

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
num_classes = 25

for epoch in np.arange(1, 21):
    model = GraphModel(num_classes)
    model.load_state_dict(torch.load(f'model_dicts/no_mars_crossers/v3_{epoch}'))
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

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Epoch: {epoch} | {accuracy}")

