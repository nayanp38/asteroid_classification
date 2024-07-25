def generate_synthetic_data(pca, num_components, num_samples):
    # Select a subset of principal components
    principal_components = pca.components_[:num_components]

    # Generate random coefficients for the selected principal components
    coefficients = np.random.normal(0, 1, size=(num_samples, num_components))

    # Reconstruct synthetic data
    synthetic_data = np.dot(coefficients, principal_components)

    return synthetic_data

def main():
    # Step 1: Perform PCA using scikit-learn
    pca = PCA()
    pca.fit(X)

    # Step 2: Decide how many principal components to retain
    num_components = 10  # Example: retain first 10 principal components

    # Step 3: Generate synthetic data
    num_samples = 100  # Example: generate 100 synthetic samples
    synthetic_data = generate_synthetic_data(pca, num_components, num_samples)

    # Print the shape of synthetic data
    print("Shape of synthetic data:", synthetic_data.shape)

if __name__ == "__main__":
    main()
