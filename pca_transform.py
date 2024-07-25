import os
import numpy as np
from sklearn.decomposition import PCA
from graph_generator import get_filename_from_number, getType, png_to_number
import matplotlib.pyplot as plt
import random
from augment_display import display_one_augment, display_one_graph

def read_data_from_files(directory, max_length=None):
    dataset = []
    dictionary = []
    min_len = float('inf')
    for filename in os.listdir(directory):
        dictionary.append(filename)
        spectrum = get_filename_from_number(filename)
        file_path = os.path.join('smass2', spectrum)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Read the second column's values from the file
                data = [float(line.split()[1]) for line in file.readlines() if line.strip()]
                if 'augmented' in filename:
                    mu = 0
                    sigma = 0.015
                    aug_data = [y + np.random.normal(mu, sigma) for y in data]
                    data = aug_data
                if data:
                    # Ensure consistent length by truncating to the minimum length
                    if max_length is not None:
                        data = data[:max_length]  # Truncate
                    # Track the minimum length
                    min_len = min(min_len, len(data))
                    # Append each data as a new row
                    dataset.append(data)
    # Truncate all pieces of data to the same length as the shortest piece of data
    dataset = [row[:min_len] for row in dataset]
    return np.array(dataset), min_len, dictionary

def compute_mean(X):
    return np.mean(X, axis=0)

def center_data(X, X_mean):
    return X - X_mean

def compute_covariance_matrix(X_c):
    # Ensure X_c has at least two dimensions
    if X_c.ndim == 1:
        X_c = np.expand_dims(X_c, axis=0)
    return np.cov(X_c.T)

def perform_pca(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

def generate_synthetic_data(pca, num_components, num_samples):
    # Select a subset of principal components
    principal_components = pca.components_[:num_components]

    # Generate random coefficients for the selected principal components
    coefficients = np.random.normal(0, 1, size=(num_samples, num_components))

    # Reconstruct synthetic data
    synthetic_data = np.dot(coefficients, principal_components)

    return synthetic_data


def augment_data_with_pca_noise(data, pca, num_samples, curr_file_num, noise_factor=0.1):
    augmented_data = []
    while len(augmented_data) < (num_samples - curr_file_num):
        # Decompose data into principal components
        data_pca = pca.transform(data)

        # Compute noise variance based on eigenvalues
        noise_variance = np.diag(pca.explained_variance_)

        # Generate random noise based on eigenvalues
        print(noise_variance.shape)
        noise = np.random.multivariate_normal(mean=np.zeros(data_pca.shape[1]), cov=noise_variance, size=len(data))

        # Augment data by adding noise
        augmented_data_pca = data_pca + noise_factor * noise

        # Reconstruct augmented data
        augmented_data.append(pca.inverse_transform(augmented_data_pca))

    # Concatenate all augmented datasets into one
    augmented_data = np.concatenate(augmented_data, axis=0) if augmented_data else None

    # Pare down the dataset to the desired number of samples
    return augmented_data[:(num_samples - curr_file_num)] if augmented_data is not None else None


def create_pca_augmentations(data_root, samples):
    with open('smass2/a000015.[2]', 'r') as file: # 6847
        # Read the second column's values from the file
        wavelengths = [float(line.split()[0]) for line in file.readlines() if line.strip()]
    for directory in os.listdir(data_root):
        if len(os.listdir(os.path.join(data_root, directory))) < samples:
            X, length, dictionary = read_data_from_files(os.path.join(data_root, directory))

            pca = PCA()
            pca.fit(X)

            # Step 3: Generate synthetic data
            num_samples = samples  # Example: generate 100 synthetic samples

            noise_factor = 0.3  # Adjust noise factor as needed

            augmented_data = augment_data_with_pca_noise(X, pca, num_samples, len(os.listdir(os.path.join(data_root, directory))), noise_factor)
            if augmented_data is not None:
                for j in range(len(augmented_data)):
                    plt.plot(wavelengths[:length], augmented_data[j])
                    # Standardize the graph to have x-axis between 0.4 and 1, and y-axis between 0.5 and 1.5

                    x_min, x_max = 0.4, 1.0
                    y_min, y_max = 0.5, 1.5

                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)

                    # Remove grid marks, axes labels, and title
                    plt.grid(False)
                    plt.xticks([])  # Remove x-axis labels
                    plt.yticks([])  # Remove y-axis labels
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.title('')

                    filename = dictionary[j % len(dictionary)]

                    # Call get_number function on the filename
                    number = png_to_number(filename)

                    # Save the augmented graph with a unique filename
                    output_directory = os.path.join(data_root, directory)
                    augmented_filename = f'{number}_pca_augmented_{len(os.listdir(output_directory))}.png'
                    augmented_filepath = os.path.join(output_directory, augmented_filename)
                    plt.savefig(augmented_filepath)
                    print(f'Created: {augmented_filepath}')

                    plt.clf()
def main():

    create_pca_augmentations('gaussian_graphs', 500)

    '''
    # Step 1: Read data from text files in a directory
    directory = 'gaussian_graphs/V'
    X, length, file_dictionary = read_data_from_files(directory)

    # Step 2: Perform PCA using scikit-learn
    pca = PCA()
    pca.fit(X)

    # Print the explained variance ratio, eigenvalues, and eigenvectors
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("\nEigenvalues:")
    print(pca.singular_values_)
    print("\nEigenvectors:")
    print(pca.components_)

    # Step 2: Decide how many principal components to retain
    num_components = 10  # Example: retain first 10 principal components

    # Step 3: Generate synthetic data
    num_samples = 500  # Example: generate 100 synthetic samples
    synthetic_data = generate_synthetic_data(pca, num_components, num_samples)

    noise_factor = 0.3  # Adjust noise factor as needed

    augmented_data = augment_data_with_pca_noise(X, pca, num_samples, 200, noise_factor)
    # Print the shape of synthetic data
    print("Shape of synthetic data:", augmented_data.shape)
    print('One sample:', augmented_data[0])

    with open('smass2/a003885.[2]', 'r') as file:
        # Read the second column's values from the file
        text_data = [float(line.split()[0]) for line in file.readlines() if line.strip()]
        print(text_data)
    print(len(text_data))
    print(len(augmented_data[0]))
    # Plot the resulting data versus the first column of the extracted text files
    plt.figure(figsize=(10, 6))
    for j in range(len(augmented_data)):
        color = plt.cm.viridis(random.random())
        plt.plot(text_data[:length], augmented_data[j], alpha=0.5, linewidth=0.5, color=color)
    x_min, x_max = 0.4, 1.0
    y_min, y_max = 0.5, 1.5

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    label_root = 'gaussian_graphs/V'

    for i in range(42):
        display_one_graph(label_root)

    for i in range(158):
        display_one_augment(label_root)

    # plt.savefig('bigger_v_500_pca.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for img in os.listdir('gaussian_graphs/A'):
        spectrum = get_filename_from_number(img)
        file_pth = os.path.join('smass2', spectrum)

        with open(file_pth, 'r') as file:
            data = [line.split() for line in file.readlines()]

        x_values = [float(row[0]) for row in data]
        y_values = [float(row[1]) for row in data]
        color = plt.cm.viridis(random.random())
        plt.plot(x_values, y_values, alpha=1, linewidth=0.5, color=color)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove grid marks, axes labels, and title
    plt.grid(False)
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.show()
    '''


if __name__ == "__main__":
    main()
