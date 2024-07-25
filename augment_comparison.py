import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images_side_by_side(original_path, augmented_path, pca_path):
    # Load the images
    original_image = mpimg.imread(original_path)
    augmented_image = mpimg.imread(augmented_path)
    pca_image = mpimg.imread(pca_path)

    # Create a larger figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # Increase the figure size

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].set_title('Without Augmented Data')
    axs[0].axis('off')

    # Display the augmented image
    axs[1].imshow(augmented_image)
    axs[1].set_title('With Gaussian Randomization Until 200 Images')
    axs[1].axis('off')

    # Display the PCA augmented image
    axs[2].imshow(pca_image)
    axs[2].set_title('With PCA Augmentation Until 500 Images')
    axs[2].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot as a larger image
    plt.savefig('bigger_comparison_plot.png', dpi=300)

    # Show the plot
    plt.show()

# Replace 'path/to/original/image', 'path/to/augmented/image', and 'path/to/pca/image'
# with the actual paths to your images
original_image_path = 'bigger_v_no_augments.png'
augmented_image_path = 'bigger_v_200_gaussian.png'
pca_image_path = 'bigger_v_500_pca.png'
display_images_side_by_side(original_image_path, augmented_image_path, pca_image_path)