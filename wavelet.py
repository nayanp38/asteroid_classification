import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pywt
import os
from matplotlib.colors import LightSource
from matplotlib import cbook, cm
from visnir_graph_generator import get_filename_from_number

# Generate some sample data (x and y values)

def cwt(x, y):
    # Perform Continuous Wavelet Transform (CWT)
    wavelet = 'cmor1.5-1'  # Choose the wavelet, here we use complex Morlet wavelet
    scales = np.arange(1, 128)  # Range of scales for wavelet analysis
    coefficients, frequencies = pywt.cwt(y, scales, wavelet)

    # Plot original signal and its CWT
    plt.figure(figsize=(12, 6))

    # Plot original signal
    # Plot original signal
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.title('Original Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Normalized Reflectance')
    plt.xlim(0.435, 0.925)

    # Plot CWT
    plt.subplot(1, 2, 2)
    im = plt.imshow(np.abs(coefficients), aspect='auto', extent=[x[0], x[-1], scales[-1], scales[0]], cmap='jet')
    plt.title('Continuous Wavelet Transform (CWT) of Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Scale')

    # Create colorbar axis
    # plt.colorbar(im, cax=cbar_ax, label='Magnitude')

    plt.tight_layout()
    plt.savefig('model_in')
    plt.show()


def cwt3d(x, y):
    # wavelet = 'cmor0.1-1'  # Choose the wavelet, here we use complex Morlet wavelet
    center = np.mean(y)
    bandwidth = np.ptp(y)
    print(bandwidth)
    bandwidth = 3.5

    wavelet = f'cmor{bandwidth}-{center}'
    scales = np.arange(1, 128)  # Range of scales for wavelet analysis
    coefficients, frequencies = pywt.cwt(y, scales, wavelet)
    print(frequencies)

    # Plot original signal and its CWT
    fig = plt.figure(figsize=(12, 6))

    # Plot original signal
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, y)
    # ax1.plot(x, (0.1*np.cos(x)) + 1)

    scale_plot = 1
    freq = frequencies[scale_plot - 2]
    period = (2 * np.pi) / freq

    in_x = [item * period for item in x]
    # ax1.plot(x, (0.1* np.cos(in_x)+1))
    ax1.set_title('Original Spectrum')
    ax1.set_xlabel('Wavelength')
    ax1.set_ylabel('Normalized Reflectance')

    # Define parameters for the Morlet wavelet
    omega_0 = 6
    sigma = 1
    # Define the range of x values
    mor_x = np.linspace(-10, 10, 1000)

    # Calculate the Morlet wavelet function
    morlet_wavelet = morlet(mor_x, omega_0, sigma, 0.75)

    # Plot the Morlet wavelet function
    # ax1.plot(mor_x, morlet_wavelet, color='green', linewidth=2)

    ax1.set_xlim(0.4, 2.5)
    ax1.set_ylim(0.5, 1.5)

    # Plot CWT as a 3D surface plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(x, scales)
    ax2.plot_surface(X, Y, np.abs(coefficients), cmap='jet')
    ax2.set_title('Continuous Wavelet Transform (CWT) of Spectrum')
    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel('Scale')
    ax2.set_zlabel('Magnitude')

    # ax2.set_zlim(0, 1.2)
    plt.tight_layout()
    plt.savefig('3d_wav')
    plt.show()


def test(x, y):
    # wavelet = 'cmor0.1-1'  # Choose the wavelet, here we use complex Morlet wavelet
    center = np.mean(y)
    bandwidth = np.ptp(y)
    print(bandwidth)

    wavelet = f'cmor{bandwidth}-{center}'
    scales = np.arange(1, 128)  # Range of scales for wavelet analysis
    coefficients, frequencies = pywt.cwt(y, scales, wavelet)
    print(frequencies)

    # Plot original signal and its CWT
    fig = plt.figure(figsize=(12, 6))

    # Plot original signal
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, y)
    # ax1.plot(x, (0.1*np.cos(x)) + 1)

    scale_plot = 1
    freq = frequencies[scale_plot - 2]
    period = (2 * np.pi) / freq

    in_x = [item * period for item in x]
    # ax1.plot(x, (0.1* np.cos(in_x)+1))
    ax1.set_title('Original Spectrum')
    ax1.set_xlabel('Wavelength')
    ax1.set_ylabel('Normalized Reflectance')

    # Define parameters for the Morlet wavelet
    omega_0 = 6
    sigma = 1
    # Define the range of x values
    mor_x = np.linspace(-10, 10, 1000)

    region = np.s_[5:50, 5:50]
    x, y, z = x[region], y[region], np.abs(coefficients)[region]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)

    plt.show()


def dwt(x, y):
    # Load spectral data (replace 'filename' with your actual filename)

    # Perform Discrete Wavelet Transform (DWT)
    wavelet = 'haar'  # Choose the wavelet, here we use Haar wavelet
    level = 5  # Choose the decomposition level
    coefficients = pywt.wavedec(y, wavelet, level=level)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.title('Original Spectral Data')
    plt.xlabel('Wavelength')
    plt.ylabel('Normalized Reflectance')

    # Plot DWT coefficients
    # Plot DWT coefficients for each level
    for i in range(level):
        plt.figure(figsize=(8, 4))
        plt.plot(coefficients[i])
        plt.title(f'DWT Coefficients (Level {i + 1})')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.show()


def morlet(x, omega_0=6, sigma=2, shift=1.0):
    """
    Morlet wavelet function.

    Parameters:
        x (array_like): Input array.
        omega_0 (float, optional): Frequency parameter. Default is 6.
        sigma (float, optional): Width parameter. Default is 2.

    Returns:
        array_like: Morlet wavelet function.
    """
    return np.exp(1j * omega_0 * (x - shift)) * np.exp(-0.5 * ((x - shift) ** 2) / sigma ** 2) + 1


def incwt(x, y):
    center = np.mean(y)
    bandwidth = np.ptp(y)

    wavelet = f'cmor{bandwidth}-{center}'
    scales = np.arange(1, 129)  # Range of scales for wavelet analysis
    coefficients, frequencies = pywt.cwt(y, scales, wavelet)

    resized_coefficients = resize_coefficients(np.abs(coefficients), 128)


    # Plot original signal and its CWT
    plt.figure(figsize=(12, 6))

    # Plot original signal
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.title('Original Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Normalized Reflectance')
    plt.xlim(0.435, 0.925)

    # Plot CWT as an image
    plt.subplot(1, 2, 2)
    plt.imshow(resized_coefficients, aspect='auto', cmap='jet', extent=[x[0], x[-1], scales[-1], scales[0]])
    plt.title('Continuous Wavelet Transform (CWT) of Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Scale')

    plt.tight_layout()
    # plt.savefig('model_in.png')
    plt.show()


    # return np.expand_dims(resized_coefficients, axis=0).astype(np.float32)


def resize_coefficients(coefficients, target_columns):
    # Determine the number of original columns
    original_columns = coefficients.shape[1]

    # Calculate the column indices for the resized array
    column_indices = np.linspace(0, original_columns - 1, target_columns).astype(int)

    # Resize the coefficients array using interpolation
    resized_coefficients = coefficients[:, column_indices]

    return resized_coefficients


def make_wavelet(png):

    with open('DeMeo2009data/5.txt', 'r') as file:
        data = [line.split() for line in file.readlines()]
    wavelength = [float(row[0]) for row in data]
    reflectance = [float(row[1]) for row in data]

    return cwt3d(wavelength, reflectance)


if __name__ == '__main__':
    make_wavelet('2.png')

    """
    with open('DeMeo2009data/4451.txt', 'r') as file:
        data = [line.split() for line in file.readlines()]

    x = [float(row[0]) for row in data]
    y = [float(row[1]) for row in data]

    cwt3d(x, y)
    """