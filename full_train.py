from model import FullModel, FullWavModel
from graph_dataset import FullDataset, FullWavDataset, DeMeoDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
from torchvision import transforms, datasets
import wandb

learning_rate = 0.001
num_epochs = 20

def one_hot_encode(index, array_size):
    # Initialize an array of zeros
    one_hot_array = [0] * array_size

    # Set 1 at the specified index
    if 0 <= index < array_size:
        one_hot_array[index] = 1
    else:
        print("Invalid index. Please provide a valid index within the array size.")

    return one_hot_array


graph_directory = 'data/cleaned_0.4'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create a dataset and dataloader
graph_dataset = DeMeoDataset(root_dir=graph_directory, transform=transform)

train_dataset, val_dataset = train_test_split(graph_dataset, test_size=0.2, random_state=42)

# Create dataloaders for training and validation
train_dataloader = DataLoader(graph_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model, loss function, and optimizer
num_classes = len(os.listdir(graph_directory))
print(f'number of asteroid types: {num_classes}')
model = FullModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    model.train()
    for images, diameter, abs_mag, albedo, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images, diameter, abs_mag, albedo)
        # one_hot = [one_hot_encode(label, 18) for label in labels]
        # one_hot = torch.tensor(one_hot).double()
        # outputs = outputs.double()
        # print(outputs)
        # print(labels)
        # outputs = outputs.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for val_images, val_diameter, val_abs_mag, val_albedo, val_labels, in val_dataloader:
            val_outputs = model(val_images, val_diameter, val_abs_mag, val_albedo)
            val_loss += criterion(val_outputs, val_labels).item()
            total_val_samples += val_labels.size(0)

    average_val_loss = val_loss / total_val_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {average_val_loss:.4f}')
    # wandb.log({"Val Loss": average_val_loss, "Loss": loss.item()})
    torch.save(model.state_dict(), f'model_dicts/demeo_aux/2_demeo+aux_{epoch+1}')

# Save the trained model
