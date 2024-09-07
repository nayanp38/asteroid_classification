import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphModel(nn.Module):
    def __init__(self, num_types):
        super(GraphModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(14400, num_types)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class GraphWavModel(nn.Module):
    def __init__(self, num_types):
        super(GraphWavModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(14400, num_types)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, image, wav):
        img_in = torch.cat([image, wav], dim=1)
        x = self.conv1(img_in)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class GraphWavAlbModel(nn.Module):
    def __init__(self, num_types):
        super(GraphWavAlbModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(14400 + 16, num_types, dtype=torch.double)
        self.fcNums = nn.Linear(1, 16, dtype=torch.double)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, image, wav, alb):
        img_in = torch.cat([image, wav], dim=1)
        x = self.conv1(img_in)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)

        albedo = alb.view(-1, 1)
        nums = self.fcNums(albedo)

        x = torch.cat([x, nums], dim=1)
        x = self.fc1(x)
        return x


class FullModel(nn.Module):
    def __init__(self, num_types):
        super(FullModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32768 + 32, num_types,
                             dtype=torch.double)  # Updated the input size to include two additional numerical inputs
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fcNums = nn.Linear(3, 32, dtype=torch.double)

    def forward(self, image, diameter, abs_mag, albedo):
        x = self.conv1(image)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)

        diameter = diameter.view(-1, 1)  # Assuming diameter and abs_mag and albedo is a scalar value
        abs_mag = abs_mag.view(-1, 1)
        albedo = albedo.view(-1, 1)

        # Concatenate the flattened image features with the numerical inputs
        nums = torch.cat([diameter, abs_mag, albedo], dim=1)
        nums = self.fcNums(nums)
        x = torch.cat([x, nums], dim=1)

        x = self.fc1(x)
        return x

    def predict_proba(self, image, diameter, abs_mag, albedo):
        # Pass the inputs through the model
        logits = self.forward(image, diameter, abs_mag, albedo)
        # Apply softmax activation to get probabilities
        probabilities = F.softmax(logits, dim=1)
        return probabilities


class FullWavModel(nn.Module):
    def __init__(self, num_types):
        super(FullWavModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32768 + 32, num_types,
                             dtype=torch.double)  # Updated the input size to include two additional numerical inputs
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fcNums = nn.Linear(3, 32, dtype=torch.double)

    def forward(self, image, diameter, abs_mag, albedo, wav):
        img_in = torch.cat([image, wav], dim=0)
        x = self.conv1(img_in)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)

        diameter = diameter.view(-1, 1)  # Assuming diameter and abs_mag and albedo is a scalar value
        abs_mag = abs_mag.view(-1, 1)
        albedo = albedo.view(-1, 1)

        # Concatenate the flattened image features with the numerical inputs
        nums = torch.cat([diameter, abs_mag, albedo], dim=1)
        nums = self.fcNums(nums)
        x = torch.cat([x, nums], dim=1)

        x = self.fc1(x)
        return x

    def predict_proba(self, image, diameter, abs_mag, albedo):
        # Pass the inputs through the model
        logits = self.forward(image, diameter, abs_mag, albedo)
        # Apply softmax activation to get probabilities
        probabilities = F.softmax(logits, dim=1)
        return probabilities
