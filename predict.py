import torch
from torchvision import transforms
from PIL import Image
from model import GraphModel
from graph_dataset import GraphDataset


# Function to preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    return image

# Load the trained model
model = GraphModel(24)  # Initialize the model with the correct number of classes
model.load_state_dict(torch.load('model_dicts/3conv_0.4_collapsed_20'))
model.eval()

# Provide the path to the image you want to predict
image_path = 'test_dir/sq_png.png'

# Preprocess the input image
input_image = preprocess_image(image_path)

# Make a prediction
with torch.no_grad():
    output = model(input_image)

# Get the predicted class index
predicted_class_idx = torch.argmax(output).item()
print(output)
print(predicted_class_idx)


graph_dataset = GraphDataset(root_dir='data/collapsed_0.4_from_avg')
predicted_class = graph_dataset.get_class_from_idx(predicted_class_idx)


# Map the class index to the class name using the class_to_idx mapping

print(f"The predicted class is: {predicted_class}")