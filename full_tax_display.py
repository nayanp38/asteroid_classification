import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib import gridspec
from matplotlib.image import imread

def plot_images_in_subdirectories(main_directory):
    # Get a list of all subdirectories in the main directory
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    # Create a grid layout for plotting
    num_subdirectories = len(subdirectories)
    num_cols = 3  # You can adjust the number of columns as needed
    num_rows = (num_subdirectories + num_cols - 1) // num_cols

    # Set up the plot
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.1, hspace=0.3)

    for i, subdir in enumerate(subdirectories):
        # Get the first png file in each subdirectory
        png_files = [f for f in os.listdir(os.path.join(main_directory, subdir)) if f.endswith('.png')]

        if png_files:
            # Take the first png file
            png_file = os.path.join(main_directory, subdir, png_files[0])

            # Load the image
            img = imread(png_file)

            # Create a subplot and plot the image
            ax = plt.subplot(gs[i])
            ax.imshow(img)
            ax.set_title(subdir, y=-0.2)
            ax.axis('off')
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())

    # Show the plot
    plt.savefig('all_classifications.png')  # Save the temporary graph for download
    plt.show()

main_directory_path = 'graphs'
plot_images_in_subdirectories(main_directory_path)
