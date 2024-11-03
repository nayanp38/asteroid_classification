from flask import Flask, render_template, request, jsonify
import rocks
from model import FullModel
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

root = "C:/Users/Nayan_Patel/PycharmProjects/asteroid_classification/"

app = Flask(__name__)
spec_data_root = 'C:/Users/Nayan_Patel/PycharmProjects/asteroid_classification/smass2'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def classify_asteroid(graph_path, diameter, abs_mag, albedo):
    model = FullModel(18)
    model.load_state_dict(torch.load(root+'model_dicts/500_model/full_model_13'))
    model.eval()
    spec_graph = Image.open('resized_plot.png')
    spec_graph = transform(spec_graph)

    diameter = torch.tensor(diameter).unsqueeze(0).double()
    abs_mag = torch.tensor(abs_mag).unsqueeze(0).double()
    albedo = torch.tensor(albedo).unsqueeze(0).double()
    spec_graph = torch.tensor(spec_graph).unsqueeze(0)

    with torch.no_grad():
        outputs = model(spec_graph, diameter, abs_mag, albedo)

    _, predicted = torch.max(outputs, 1)

    class_dict = ['A', 'B', 'C', 'Cg', 'Ch', 'D', 'E', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S', 'V', 'X', 'Z']

    return class_dict[predicted]


def get_spec_graph(number, name):
    filename = f'a{number:06d}.[2]'
    file_path = os.path.join(spec_data_root, filename)
    with open(file_path, 'r') as file:
        data = [line.split() for line in file.readlines()]

    x_values = [float(row[0]) for row in data]
    y_values = [float(row[1]) for row in data]

    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5
    x_min, x_max = 0.4, 1.0
    y_min, y_max = 0.5, 1.5

    plt.plot(x_values, y_values)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks(np.arange(0.4, 1.1, 0.1))  # Remove x-axis labels
    plt.yticks(np.arange(0.5, 1.6, 0.1))  # Remove y-axis labels
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Reflectance')
    if name[-1] == 's':
        plt.title(f"{name}' Spectrum")
    else:
        plt.title(f"{name}'s Spectrum")

    display_image_path = 'static/display_img.png'
    plt.savefig(display_image_path, format='png', dpi=300)

    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    image_path = 'temp_plot.png'
    plt.savefig(image_path, format='png', dpi=300)  # Save as high resolution

    # Close the plot to free memory
    plt.close()

    # Open the saved image and resize it to 128x128 pixels
    image = Image.open(image_path).convert('L')  # 'L' mode converts image to grayscale

    # Save the resized image
    resized_image_path = 'resized_plot.png'
    image.save(resized_image_path)

    # Return the path to the resized image
    return resized_image_path

# Function to format numbers into three-column format
def format_numbers(numbers_str):
    numbers_list = [float(numbers_str[i:i+5]) for i in range(0, len(numbers_str), 5)]
    formatted_numbers = [f"{num:.4f}" for num in numbers_list]

    # Splitting into three-column format
    formatted_content = [formatted_numbers[i:i+3] for i in range(0, len(formatted_numbers), 3)]
    formatted_text = '\n'.join(['\t'.join(row) for row in formatted_content])

    print(formatted_text)

    with open("formatted_numbers.txt", "w") as file:
        file.write(formatted_text)


# Function to classify the asteroid based on its number
def get_aux_data(number):
    try:
        rock = rocks.Rock(number)
    except:
        return None
    return (rock.diameter.value,
            rock.absolute_magnitude.value,
            rock.albedo.value)

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/spectrum', methods=['GET'])
def spectrum():
    try:
        inspec = request.args.get('spectrum')
        print(inspec)
        format_numbers(inspec)
        diameter = request.args.get('diameter')
        abs_mag = request.args.get('abs_mag')
        albedo = request.args.get('albedo')
        number = rocks.Rock(id).number
        name = rocks.Rock(id).name
        image = Image.open(get_spec_graph(number, name))
        diam, abs_mag, alb = get_aux_data(number)
        classification = classify_asteroid(image, diam, abs_mag, alb)
        classification = 'Type: ' + classification
        aux_data = ('Diameter: ' + str(diam) + ' km' +
                    '\nAbsolute Magnitude: ' + str(abs_mag) +
                    '\nAlbedo: ' + str(alb))
        identity = ('Name: ' + str(name) + '\nNumber: ' + str(number))
        return jsonify({
            'classification': classification,
            'aux_data': aux_data,
            'identity': identity})
    except:
        return jsonify({'classification': 'One or more fields are blank!', 'aux_data': '', 'identity':''})

# Route to handle the classification request
@app.route('/classify', methods=['GET'])
def classify():
    try:
        id = int(request.args.get('id'))  # Try converting to integer
    except ValueError:
        id = request.args.get('id') # if not an integer, that means try to search by asteroid name

    number = rocks.Rock(id).number
    if number is None: # if number is none, the rock doesn't exist
         return jsonify({'classification': 'Asteroid not found', 'aux_data': '', 'identity':''})

    name = rocks.Rock(id).name
    image = Image.open(get_spec_graph(number, name))
    diam, abs_mag, alb = get_aux_data(number)
    classification = classify_asteroid(image, diam, abs_mag, alb)
    classification = 'Type: ' + classification
    aux_data = ('Diameter: ' + str(diam) + ' km' +
                '\nAbsolute Magnitude: ' + str(abs_mag) +
                '\nAlbedo: ' + str(alb))
    identity = ('Name: ' + str(name) + '\nNumber: ' + str(number))
    return jsonify({
        'classification': classification,
        'aux_data': aux_data,
        'identity': identity})


if __name__ == '__main__':
    app.run(debug=True)
