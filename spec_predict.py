import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from spec_model import AsteroidModel  # Assuming the model is saved in a file named 'model.py'

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

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)

input_size = X_train.shape[1]
output_size = y_train.shape[1]

model = AsteroidModel(input_size, 64, output_size)
model.load_state_dict(torch.load('model_dicts/v1.pth'))  # Replace with the actual path to your saved model

# Set the model to evaluation mode
model.eval()

# Choose one piece of data for testing
sample_index = 0
sample_data = X_test_tensor[sample_index].unsqueeze(0)  # Add a batch dimension

# Make a prediction
with torch.no_grad():
    predicted_output = model(sample_data)
    _, predicted_label = torch.max(predicted_output, 1)

# Convert predicted label back to the original class name

# Print the result
print(f'True Label: {y_test[sample_index]}')
print(f'Predicted Label: {predicted_output}')