import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Generate some sample data (x and y values)
# Replace this with your actual data
with open('smass2/a000004.[2]', 'r') as file:
    data = [line.split() for line in file.readlines()]

x = [float(row[0]) for row in data]
y = [float(row[1]) for row in data]

# Perform Fast Fourier Transform (FFT)
fft_output = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(y), d=(x[1] - x[0]))  # Calculate the frequencies corresponding to FFT output

N = 600
T = 1.0/800.0
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
# Plot original signal and its FFT
plt.figure(figsize=(12, 6))

# Plot original signal
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot FFT
plt.subplot(2, 1, 2)
# plt.plot(frequencies, np.abs(fft_output))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title('FFT of Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
