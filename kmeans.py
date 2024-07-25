import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Define K-Means model
class KMeans(nn.Module):
    def __init__(self, num_clusters, num_features):
        super(KMeans, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_clusters, num_features))

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        _, assignments = torch.min(distances, dim=1)
        return assignments


# Define FeatureExtractor model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


# Function to extract features from images using the FeatureExtractor model
def extract_features_neural_network(image_path, feature_extractor):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature_vector = feature_extractor(image)
    return feature_vector.squeeze().numpy()


# Load and preprocess images
def load_images(directory, feature_extractor):
    data = []
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            image_path = os.path.join(directory, os.path.join(folder, filename))
            print(image_path)
            feature_vector = extract_features_neural_network(image_path, feature_extractor)
            data.append(feature_vector)

    return np.array(data)


# Set up FeatureExtractor model
feature_extractor_model = FeatureExtractor()

# Set up K-Means model
num_clusters = 5
num_features = 20000  # Adjust based on the number of features extracted by the neural network
kmeans_model = KMeans(num_clusters, num_features)

# Load and preprocess images
image_directory = 'graphs'
image_data = load_images(image_directory, feature_extractor_model)

# Standardize data
scaler = StandardScaler()
image_data_scaled = scaler.fit_transform(image_data.reshape(-1, 1))

# Convert to PyTorch tensor
image_tensor = torch.tensor(image_data_scaled, dtype=torch.float32)

# Define loss function
criterion = nn.MSELoss()

# Set up optimizer
optimizer = torch.optim.SGD(kmeans_model.parameters(), lr=0.01)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    # Forward pass
    assignments = kmeans_model(image_tensor)

    # Calculate loss
    loss = criterion(image_tensor, kmeans_model.centers[assignments])

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

torch.save(kmeans_model.state_dict(), 'model_dicts/full_model_v3_aug+mass.pth')


def visualize_clusters(image_directory, assignments, num_clusters=3):
    # Create subplots
    fig, axs = plt.subplots(1, num_clusters, figsize=(15, 5))

    for cluster_idx in range(num_clusters):
        # Find images representative of the cluster
        cluster_images = [os.path.join(image_directory, filename) for filename, cluster_id in
                          zip(os.listdir(image_directory), assignments.numpy()) if cluster_id == cluster_idx]

        if cluster_images:
            # Display the first representative image
            representative_image = cluster_images[0]
            img = Image.open(representative_image)
            axs[cluster_idx].imshow(img)
            axs[cluster_idx].axis('off')
            axs[cluster_idx].set_title(f'Cluster {cluster_idx}')
        else:
            # If no images in the cluster, display a placeholder
            axs[cluster_idx].axis('off')
            axs[cluster_idx].set_title(f'Cluster {cluster_idx}\n(No images)')

    plt.show()


assignments_after_training = kmeans_model(image_tensor)
visualize_clusters(image_directory, assignments_after_training)
