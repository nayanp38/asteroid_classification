from model import GraphModel, GraphWavModel
from graph_dataset import GraphDataset, GraphWavDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from PIL import Image
from torchvision import transforms, datasets
import wandb
from sklearn.metrics import accuracy_score


learning_rate = 0.001
num_epochs = 20
'''
run = wandb.init(
    # Set the logging location
    project="asteroid",
    # Track params
    config={
        "learning_rate": 0.01,
        "epochs": num_epochs,
    },
)

'''

deviation = 0.4
graph_directory = f'C:/Users/Nayan_Patel/PycharmProjects/asteroid/visnir_graphs_{deviation}_from_avg'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a dataset and dataloader
graph_dataset = GraphDataset(root_dir=graph_directory, transform=transform)

train_dataset, val_dataset = train_test_split(graph_dataset, test_size=0.2, random_state=42)

# Create dataloaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model, loss function, and optimizer
num_classes = len(os.listdir(graph_directory))
print(f'number of asteroid types: {num_classes}')
model = GraphModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for val_images, val_labels in val_dataloader:
            val_outputs = model(val_images)
            _, predicted = torch.max(val_outputs, 1)
            val_loss += criterion(val_outputs, val_labels).item()
            total_val_samples += val_labels.size(0)
            val_acc = accuracy_score(val_labels, predicted)

    average_val_loss = val_loss / total_val_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    # wandb.log({"Val Loss": average_val_loss, "Loss": loss.item()})
    if epoch > 10:
        torch.save(model.state_dict(), f'model_dicts/new_visnir_graphs_{deviation}_from_avg_epoch{epoch+1}.pth')

# Save the trained model
