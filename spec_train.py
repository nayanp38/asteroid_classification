import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from spec_model import AsteroidModel

# Load the CSV file
csv_file_path = 'processed_csv/processed_data.csv'  # Replace with the actual path to your processed CSV file
df = pd.read_csv(csv_file_path)


# Create an array that describes the one-hot for each asteroid, then create a new
# tensor that aggregates all the one-hot values
def csv_to_tensor(df):

    # Convert true/false values to 1s and 0s
    df = df.drop(columns='Diameter (km)')
    tensor = torch.FloatTensor(df.astype(int).values)

    return tensor


# Assuming 'Diameter' is the feature and 'AsteroidType' columns are the labels
X = df[['Diameter (km)']].values.astype(float)
y = csv_to_tensor(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)

print(X_train_tensor.shape)
print(X_train_tensor[0])
print(y_train.shape)
print(y_train[0])


# Initialize the model
input_size = X_train.shape[1]
hidden_size = 64
output_size = y_train.shape[1]

model = AsteroidModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 16

for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test)

print(f'Test Loss: {test_loss.item():.4f}')

torch.save(model.state_dict(), 'model_dicts/v1.pth')