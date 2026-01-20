import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt

def complex_morlet_transform(signal, scales, wavelet='cmor'):
    """
    Perform a complex Morlet wavelet transform on the input signal.

    Parameters:
    signal (array-like): The input signal to be transformed.
    scales (array-like): The scales at which to compute the wavelet transform.
    wavelet (str): The type of wavelet to use. Default is 'cmor' for complex Morlet.

    Returns:
    tuple: A tuple containing the wavelet coefficients and frequencies.
    """

    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    return coefficients, frequencies

if __name__ == "__main__":
    # Example usage
    t = np.linspace(0, 1, 200, endpoint=False)
    signal = np.sin(2 * np.pi * 7 * t) + np.sin(2 * np.pi * 13 * t)

    scales = np.arange(1, 128)
    coefficients, frequencies = complex_morlet_transform(signal, scales)

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Input Signal')
    
    plt.subplot(2, 1, 2)

    plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto',
               vmax=abs(coefficients).max(), vmin=0)
    plt.colorbar(label='Magnitude')
    plt.ylabel('Scale')
    plt.xlabel('Time [s]')
    plt.title('Complex Morlet Wavelet Transform')
    plt.show()