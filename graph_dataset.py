from torch.utils.data import Dataset
import os
import re
import numpy as np
from PIL import Image
import torch
from graph_generator import get_abs_mag, get_diameter, png_to_number, get_albedo
from wavelet import incwt, make_wavelet


class GraphDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        print(self.classes)
        print(self.class_to_idx)
        print(self.idx_to_class)

        # Collect paths to images and their corresponding labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect paths to images in this class
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            image_paths = [os.path.join(class_dir, img) for img in image_files]

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes


class GraphWavDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        print(self.classes)
        print(self.class_to_idx)
        print(self.idx_to_class)

        # Collect paths to images and their corresponding labels
        self.samples = []
        self.wavs = []

        '''
        self.albedos = []

        with open('demeo_albedos.txt.txt', 'r') as file:
            data = [line.split() for line in file.readlines()]

        nums = [int(row[0]) for row in data]
        measured_albedos = [float(row[1]) for row in data]
        '''

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            img_list = os.listdir(class_dir)

            # Collect paths to images in this class
            image_files = [f for f in img_list if f.endswith('.png')]
            image_paths = [os.path.join(class_dir, img) for img in image_files]

            wavelet_transforms = [make_wavelet(f) for f in img_list if f.endswith('.png')]

            '''
            pattern = re.compile(r'(\d+)\.png')
            asteroid_nums = [int(pattern.match(i).group(1)) for i in img_list]
            indices = [nums.index(i) for i in asteroid_nums]
            albs = [measured_albedos[i] for i in indices]
            '''

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])
            self.wavs.extend(wavelet_transforms)
            # self.albedos.extend(albs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        wav = self.wavs[idx]
        # albedo = self.albedos[idx]
        return image, wav, label

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes



class FullDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        # Collect paths to images and their corresponding labels
        self.samples = []
        self.diameters = []  # List to store diameters
        self.abs_mags = []  # List to store absolute magnitudes
        self.albedos = []

        diameter_path = '500_diameters'

        # Open the file in read mode and read the values into a list
        with open(diameter_path, 'r') as file:
            self.diameters = [float(line.strip()) for line in file.readlines()]

        abs_mags_path = '500_abs_mags'

        # Open the file in read mode and read the values into a list
        with open(abs_mags_path, 'r') as file:
            self.abs_mags = [float(line.strip()) for line in file.readlines()]

        albedos_path = '500_albedos'

        # Open the file in read mode and read the values into a list
        with open(albedos_path, 'r') as file:
            self.albedos = [float(line.strip()) for line in file.readlines()]


        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect paths to images in this class
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            image_paths = [os.path.join(class_dir, img) for img in image_files]

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])

            '''
            # Get diameters and append to the diameters list
            class_diameters = self.get_diameters(image_files)
            self.diameters.extend(class_diameters)

            # Get absolute magnitudes and append to the abs_mags list
            class_abs_mags = self.get_abs_mags(image_files)
            self.abs_mags.extend(class_abs_mags)
            
            class_albedos = self.get_albedos(image_files)
            self.albedos.extend(class_albedos)
            '''


        '''
        file_path = "500_diameters"

        # Open the file in write mode and save the list
        with open(file_path, 'w') as file:
            for item in self.diameters:
                file.write(f"{item}\n")

        print(f"List saved to: {file_path}")

        file_path = "500_abs_mags"

        # Open the file in write mode and save the list
        with open(file_path, 'w') as file:
            for item in self.abs_mags:
                file.write(f"{item}\n")

        print(f"List saved to: {file_path}")
        
        # Convert diameters and abs_mags lists to torch tensors
        self.diameters = torch.tensor(self.diameters, dtype=torch.float32)
        self.abs_mags = torch.tensor(self.abs_mags, dtype=torch.float32)
        
        file_path = "500_albedos"

        # Open the file in write mode and save the list
        with open(file_path, 'w') as file:
            for item in self.albedos:
                file.write(f"{item}\n")

        print(f"List saved to: {file_path}")
        '''

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # Get the corresponding diameter and absolute magnitude and albedo for this image
        diameter = self.diameters[idx]
        abs_mag = self.abs_mags[idx]
        albedo = self.albedos[idx]

        return image, diameter, abs_mag, albedo, label

    def get_albedos(self, image_files):
        albedos = [get_albedo(img) for img in image_files]

        return albedos

    def get_diameters(self, image_files):
        # Placeholder for the get_diameter() function, replace with your actual implementation
        diameters = [get_diameter(img) for img in image_files]

        return diameters

    def get_abs_mags(self, image_files):
        # Placeholder for the get_abs_mag() function, replace with your actual implementation
        abs_mags = [get_abs_mag(img) for img in image_files]

        return abs_mags

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes


class Original200Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        # Collect paths to images and their corresponding labels
        self.samples = []
        self.diameters = []  # List to store diameters
        self.abs_mags = []  # List to store absolute magnitudes
        self.albedos = []

        diameter_path = 'diameters'

        # Open the file in read mode and read the values into a list
        with open(diameter_path, 'r') as file:
            self.diameters = [float(line.strip()) for line in file.readlines()]

        abs_mags_path = 'abs_mags'

        # Open the file in read mode and read the values into a list
        with open(abs_mags_path, 'r') as file:
            self.abs_mags = [float(line.strip()) for line in file.readlines()]

        albedos_path = 'albedos'

        # Open the file in read mode and read the values into a list
        with open(albedos_path, 'r') as file:
            self.albedos = [float(line.strip()) for line in file.readlines()]

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect paths to images in this class
            self.image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            image_paths = [os.path.join(class_dir, img) for img in self.image_files]

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])

            '''
            # Get diameters and append to the diameters list
            class_diameters = self.get_diameters(image_files)
            self.diameters.extend(class_diameters)

            # Get absolute magnitudes and append to the abs_mags list
            class_abs_mags = self.get_abs_mags(image_files)
            self.abs_mags.extend(class_abs_mags)

            class_albedos = self.get_albedos(image_files)
            self.albedos.extend(class_albedos)
            '''

        '''
        file_path = "500_diameters"

        # Open the file in write mode and save the list
        with open(file_path, 'w') as file:
            for item in self.diameters:
                file.write(f"{item}\n")

        print(f"List saved to: {file_path}")

        file_path = "500_abs_mags"

        # Open the file in write mode and save the list
        with open(file_path, 'w') as file:
            for item in self.abs_mags:
                file.write(f"{item}\n")

        print(f"List saved to: {file_path}")

        # Convert diameters and abs_mags lists to torch tensors
        self.diameters = torch.tensor(self.diameters, dtype=torch.float32)
        self.abs_mags = torch.tensor(self.abs_mags, dtype=torch.float32)

        file_path = "500_albedos"

        # Open the file in write mode and save the list
        with open(file_path, 'w') as file:
            for item in self.albedos:
                file.write(f"{item}\n")

        print(f"List saved to: {file_path}")
        '''

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # Get the corresponding diameter and absolute magnitude and albedo for this image
        diameter = self.diameters[idx]
        abs_mag = self.abs_mags[idx]
        albedo = self.albedos[idx]
        # img = self.image_files[idx]

        return image, diameter, abs_mag, albedo, label

    def get_albedos(self, image_files):
        albedos = [get_albedo(img) for img in image_files]

        return albedos

    def get_diameters(self, image_files):
        # Placeholder for the get_diameter() function, replace with your actual implementation
        diameters = [get_diameter(img) for img in image_files]

        return diameters

    def get_abs_mags(self, image_files):
        # Placeholder for the get_abs_mag() function, replace with your actual implementation
        abs_mags = [get_abs_mag(img) for img in image_files]

        return abs_mags

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes


class FullWavDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        # Collect paths to images and their corresponding labels
        self.samples = []
        self.diameters = []  # List to store diameters
        self.abs_mags = []  # List to store absolute magnitudes
        self.albedos = []
        self.wavs = []

        diameter_path = '500_diameters'

        # Open the file in read mode and read the values into a list
        with open(diameter_path, 'r') as file:
            self.diameters = [float(line.strip()) for line in file.readlines()]

        abs_mags_path = '500_abs_mags'

        # Open the file in read mode and read the values into a list
        with open(abs_mags_path, 'r') as file:
            self.abs_mags = [float(line.strip()) for line in file.readlines()]

        albedos_path = '500_albedos'

        # Open the file in read mode and read the values into a list
        with open(albedos_path, 'r') as file:
            self.albedos = [float(line.strip()) for line in file.readlines()]

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect paths to images in this class
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            image_paths = [os.path.join(class_dir, img) for img in image_files]

            wavelet_transforms = [make_wavelet(f) for f in os.listdir(class_dir) if f.endswith('.png')]

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])
            self.wavs.extend(wavelet_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # Get the corresponding diameter and absolute magnitude and albedo for this image
        diameter = self.diameters[idx]
        abs_mag = self.abs_mags[idx]
        albedo = self.albedos[idx]
        wav = self.wavs[idx]

        return image, diameter, abs_mag, albedo, label, wav

    def get_albedos(self, image_files):
        albedos = [get_albedo(img) for img in image_files]

        return albedos

    def get_diameters(self, image_files):
        # Placeholder for the get_diameter() function, replace with your actual implementation
        diameters = [get_diameter(img) for img in image_files]

        return diameters

    def get_abs_mags(self, image_files):
        # Placeholder for the get_abs_mag() function, replace with your actual implementation
        abs_mags = [get_abs_mag(img) for img in image_files]

        return abs_mags

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes


class DeMeoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        # Collect paths to images and their corresponding labels
        self.samples = []
        self.diameters = []  # List to store diameters
        self.abs_mags = []  # List to store absolute magnitudes
        self.albedos = []


        for class_name in self.classes:
            self.image_files = []
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect paths to images in this class
            for img in os.listdir(class_dir):
                self.image_files += [img]
                number = img[:-4]

                with open('data/demeo_aux.txt', 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts[0] == number:
                            self.abs_mags += [float(parts[1])]
                            self.diameters += [float(parts[2])]
                            self.albedos += [float(parts[3])]
            image_paths = [os.path.join(class_dir, img) for img in self.image_files]

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # Get the corresponding diameter and absolute magnitude and albedo for this image
        diameter = self.diameters[idx]
        abs_mag = self.abs_mags[idx]
        albedo = self.albedos[idx]
        # img = self.image_files[idx]

        return image, diameter, abs_mag, albedo, label

    def get_albedos(self, image_files):
        albedos = [get_albedo(img) for img in image_files]

        return albedos

    def get_diameters(self, image_files):
        diameters = [get_diameter(img) for img in image_files]

        return diameters

    def get_abs_mags(self, image_files):
        abs_mags = [get_abs_mag(img) for img in image_files]

        return abs_mags

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all subdirectories (each subdirectory is a class)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        print(self.classes)
        print(self.class_to_idx)
        print(self.idx_to_class)

        # Collect paths to images and their corresponding labels
        self.samples = []
        self.numbers = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Collect paths to images in this class
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            image_paths = [os.path.join(class_dir, img) for img in image_files]
            numbers = [f[:-4] for f in os.listdir(class_dir) if f.endswith('.png')]

            # Append (image path, class index) tuples to the samples list
            self.samples.extend([(path, class_idx) for path in image_paths])
            self.numbers.extend(numbers)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        number = self.numbers[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, number

    def get_class_from_idx(self, idx):
        return self.idx_to_class[idx]

    def get_classes(self):
        return self.classes
