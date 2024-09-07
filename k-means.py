import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Function to load and preprocess images
def load_images_from_directory(directory, target_size=(128, 128)):
    images = []
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=target_size, color_mode="grayscale")
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            image_paths.append(img_path)
    return np.array(images), image_paths


# Load your own dataset
dataset_directory = 'C:/Users/Nayan_Patel/OneDrive - Cary Academy/python_packups/data/visnir_graphs_clustering'
x_train, image_paths = load_images_from_directory(dataset_directory, (128, 128))
x_train = np.expand_dims(x_train, axis=-1)

# Define the convolutional autoencoder
input_img = Input(shape=(128, 128, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=4,
                shuffle=True)

# Encode the images to lower-dimensional representations
encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_train)

# Flatten the encoded images
encoded_imgs_flat = encoded_imgs.reshape((len(x_train), -1))

# Apply K-Means clustering
n_clusters = 25
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(encoded_imgs_flat)
cluster_labels = kmeans.labels_

output_directory = 'C:/Users/Nayan_Patel/PycharmProjects/asteroid/cluster_output'
for cluster in range(n_clusters):
    cluster_dir = os.path.join(output_directory, f'cluster_{cluster}')
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

# Move images to the corresponding cluster directory
for img_path, cluster in zip(image_paths, cluster_labels):
    img = load_img(img_path, target_size=(128, 128), color_mode="grayscale")
    img_array = img_to_array(img)
    img = array_to_img(img_array)
    cluster_dir = os.path.join(output_directory, f'cluster_{cluster}')
    img.save(os.path.join(cluster_dir, os.path.basename(img_path)))
