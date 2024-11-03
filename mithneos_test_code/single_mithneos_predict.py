import torch
from torchvision import transforms
from PIL import Image
from model import GraphModel
from graph_dataset import GraphDataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def find_file_by_number(directory, number):
    # Construct the target file name based on the number
    target_file = f"{number}.png"

    # Iterate through the files in the specified directory
    for file_name in os.listdir(directory):
        # Check if the current file matches the target file name
        if file_name == target_file:
            file_path = os.path.join(directory, file_name)
            # print(f"File found: {file_path}")
            return file_path

    print(f"No file named {number}.png found in {directory}")
    return None

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
graph_dataset = GraphDataset(root_dir='../data/collapsed_0.4_from_avg')

dict_path = '../model_dicts/3conv_0.4_collapsed_full_trainset_20.pth'
dict = '3conv_0.4_collapsed_full_trainset_20.pth'

model.load_state_dict(torch.load(f'../model_dicts/{dict}'))
model.eval()

# Provide the path to the image you want to predict
asteroids = ['1468', '1917', '1981', '5392', '6611', '7889', '8566', '33881', '88188', '192563', '253841', '297418',
             '326290', '414586', '2001YE4'] # v class
asteroids = ['326290'] # misclassified v

asteroids = ['3552', '8373', '52762', '162998', '301964', '2000PG3'] # d class
asteroids = ['8373', '52762', '301964'] # misclassified d class

asteroids = ['2449', '3554', '3671', '3691', '4660', '5645', '10302', '14402', '24761', '33342', '52768', '54789',
             '65996', '102528', '137170', '164202', '180186', '267494', '312473', '363067', '405058', '2001SG286',
             '2002TS67', '2004QD3'] # x class

asteroids = ['1468', '1917', '1981', '5392', '6611', '7889', '8566', '33881', '88188', '192563', '253841', '297418',
             '326290', '414586', '2001YE4']


for num in asteroids:
    image_path = find_file_by_number('../data/mithneos_graphs/', str(num))
    classifications = {}
    # Preprocess the input image

    input_image = preprocess_image(image_path)

    # Make a prediction
    with torch.no_grad():
        output = model(input_image)

    # Get the predicted class index
    predicted_class_idx = torch.argmax(output).item()

    predicted_class = graph_dataset.get_class_from_idx(predicted_class_idx)
    classifications[num] = predicted_class

    true_classifications = {}

    with open('../data/MITHNEOS.txt', 'r') as file:
        text = file.read()
    for line in text.strip().split('\n'):
        # Split the line by commas and strip whitespace
        parts = [part.strip() for part in line.split('\t')]

        # Extract keys and value
        key1, key2, value = parts

        # If key1 is not blank, map both key1 and key2 to the value
        if key1:
            true_classifications[key1.replace(' ', '')] = value

        # Always map key2 to the value
        true_classifications[key2.replace(' ', '')] = value

    true_class = true_classifications[num]

    img = mpimg.imread(image_path)  # Replace 'image_path.jpg' with your image file path
    plt.imshow(img)
    plt.axis('off')  # Turn off the axis labels
    plt.show()

    print(f"{num}: Pred: {predicted_class}, True: {true_class}")