import torch
from torchvision import transforms
from PIL import Image
from model import GraphModel, FullModel
from graph_dataset import GraphDataset, DeMeoDataset
import os


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    return image


model = FullModel(24)  # Initialize the model with the correct number of classes
graph_dataset = DeMeoDataset(root_dir='data/cleaned_0.4')

dict_path = 'model_dicts/demeo_aux/demeo+aux_18'

model.load_state_dict(torch.load(dict_path))
model.eval()

# Provide the path to the image you want to predict
images_path = 'data/mithneos_graphs/'
classifications = {}
# Preprocess the input image
for image in os.listdir(images_path):
    image_path = os.path.join(images_path, image)
    input_image = preprocess_image(image_path)
    num = image[:-4]

    with open('data/mithneos_aux.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == num:
                abs_mag = float(parts[1])
                diameter = float(parts[2])
                albedo = float(parts[3])

    diameter = torch.tensor(diameter).unsqueeze(0).double()
    abs_mag = torch.tensor(abs_mag).unsqueeze(0).double()
    albedo = torch.tensor(albedo).unsqueeze(0).double()

    # Make a prediction
    with torch.no_grad():
        output = model(input_image, diameter, abs_mag, albedo)

    # Get the predicted class index
    predicted_class_idx = torch.argmax(output).item()

    predicted_class = graph_dataset.get_class_from_idx(predicted_class_idx)
    classifications[num] = predicted_class

    # Map the class index to the class name using the class_to_idx mapping

with open(f'data/demeo_predictions.txt', 'w') as file:
    for key, value in classifications.items():
        file.write(f'{key} {value}\n')
