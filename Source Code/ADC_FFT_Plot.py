import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, correlate, hamming

# Load dataset from CSV into a DataFrame, selecting specific columns
df = pd.read_csv(r'D:\individualProject\Dataset\adc_aluminiumfoil_1m.csv')
df = df.iloc[:, 16:]

# Initialize an array to store peak data (not used in this snippet)
peak = np.zeros((len(df.index),), dtype=int)

# Define the specific frequency range in kHz
range_min, range_max = 30, 50

# Iterate over each row in the DataFrame (each signal)
for i in range(len(df.index)):
    f = df.iloc[i, :]  # Extract the signal from the current row

    # Signal processing parameters
    n = len(f)  # Length of the signal
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs
    time = np.arange(n) / Fs  # Time array for plotting, adjusted to seconds

    # Fourier Transform and Power Spectral Density (PSD) calculation
    fhat = np.fft.fft(f, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n) / 1000  # Convert frequency to kHz
    L = np.arange(1, n // 2, dtype='int')

    # Focus on the desired frequency range (30-50 kHz)
    freq_range_mask = (freq >= range_min) & (freq <= range_max)
    L_range = L[freq_range_mask[L]]

    # Filter the signal by setting a threshold on the PSD
    indices = PSD > 1.5  # Threshold for filtering
    PSDclean = PSD * indices  # Filtered PSD
    fhat = indices * fhat  # Filtered Fourier Transform
    ffilt = np.fft.ifft(fhat)  # Inverse FFT to get the filtered signal

    # Hilbert Transform to compute the analytical signal and its envelope
    analytical_signal = hilbert(ffilt.real)
    env = np.abs(analytical_signal)  # Envelope of the analytical signal

    # Peak detection in the envelope
    x, _ = find_peaks(env, distance=n)

    # Compute the autocorrelation of the filtered signal
    autocorr_signal = correlate(ffilt.real, ffilt.real, mode='full')
    autocorr_signal = autocorr_signal[autocorr_signal.size // 2:]

    # Apply a Hamming window to the autocorrelated signal
    window = hamming(len(autocorr_signal))
    windowed_signal = autocorr_signal * window

    # Perform the FFT on the windowed autocorrelated signal
    fft_windowed = np.fft.fft(windowed_signal)
    PSD_windowed = fft_windowed * np.conj(fft_windowed) / len(windowed_signal)
    freq_windowed = (1 / (dt * len(windowed_signal))) * np.arange(len(fft_windowed)) / 1000  # Convert to kHz
    L_windowed = np.arange(1, len(fft_windowed) // 2, dtype='int')

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    # fig, axs = plt.subplots(4, 1, figsize=(10,20))

    # Plot 1: Original Noisy Signal
    axs[0].plot(time, f, label='Noisy')
    axs[0].set_xlim(time[0], time[-1])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_title('Original Noisy Signal')

    # Plot 2: Filtered Signal with Envelope and Peaks
    axs[1].plot(time, ffilt.real, label='Filtered Signal')
    axs[1].plot(time, env, label='Envelope')
    axs[1].plot(time[x], env[x], "x", label='Peaks', color='red')
    axs[1].set_xlim(time[0], time[-1])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[1].set_title('Filtered Signal with Envelope and Peaks')

    # Plot 3: FFT of Signal (30-50 kHz)
    axs[2].plot(freq[L_range], np.abs(PSD[L_range]), color='c', linewidth=1)
    axs[2].plot(freq[L_range], np.abs(PSDclean[L_range]), color='k', linewidth=1.5, label='Filtered')
    axs[2].set_xlim(range_min, range_max)
    axs[2].set_xlabel('Frequency (kHz)')
    axs[2].set_ylabel('Power')
    axs[2].legend()
    axs[2].set_title('FFT of Signal (30-50 kHz)')

    # # Plot 4: FFT of Windowed Autocorrelated Signal
    # axs[3].plot(freq_windowed[L_windowed], np.abs(PSD_windowed[L_windowed]), color='b', linewidth=1.5)
    # axs[3].set_xlim(range_min, range_max)
    # axs[3].set_xlabel('Frequency (kHz)')
    # axs[3].set_ylabel('Power')
    # axs[3].set_title('FFT of Windowed Autocorrelated Signal')

    plt.tight_layout()
    plt.show()

    # Stop after the first signal due to the break statement
    break
