import numpy as np
import pandas as pd
from scipy.signal import hilbert, get_window, find_peaks
import time
import pywt
from scipy.signal import correlate
def read_and_prepare_data(dataset_path, labels):
    data_list = []

    for file_path, label in zip(dataset_path, labels):
        # Load the dataset with no header, assuming it doesn't have one
        dataframe = pd.read_csv(file_path, header=None)

        # Select columns starting from the 17th (index 16) onwards
        df = dataframe.iloc[:, 16:]

        # Append to list with label
        data_list.append((df, label))

    return data_list


def apply_window(signal, window_type='hann'):
    """
    Applies a windowing function to a signal to reduce spectral leakage.

    Parameters:
    - signal: ndarray, the original signal to be windowed.
    - window_type: str or tuple, specifying the type of window to apply.

    Returns:
    - The windowed signal as an array.
    """
    # Validate window type input
    if not isinstance(window_type, (str, tuple)):
        raise ValueError("Window type must be a string or a tuple")
    # Generate the window based on the specified type and signal length
    window = get_window(window_type, len(signal))
    # Apply the window to the signal by element-wise multiplication
    return signal * window

def extract_features(data_list, dt):
    all_features = []
    peaks_list_all = []

    for df, label in data_list:
        distances = np.zeros((len(df.index),), dtype=float)
        peaks_list = np.zeros((len(df.index),), dtype=int)
        filtered_signals = np.zeros(df.shape)  # Initialize array to store filtered signals

        wavelet_features = []
        autocorr_features = []

        for i in range(len(df.index)):
            f = df.iloc[i, :]

            # Apply window function to the signal to reduce edge effects
            windowed_signal = apply_window(f)

            # Wavelet Transform for time-frequency analysis
            coeffs = pywt.wavedec(windowed_signal, 'db4', level=5)
            wavelet_energy = np.sum([np.sum(np.abs(c) ** 2) for c in coeffs])
            wavelet_features.append(wavelet_energy)

            # Fourier Transform for frequency analysis
            n = len(windowed_signal)
            fhat = np.fft.fft(windowed_signal, n)
            PSD = fhat * np.conj(fhat) / n

            indices = PSD > 1.5  # Thresholding the Power Spectral Density
            fhat = indices * fhat
            ffilt = np.fft.ifft(fhat)  # Inverse FFT for filtered signal

            filtered_signals[i, :] = ffilt.real  # Store the filtered signal

            # Autocorrelation to find periodic components
            autocorr = correlate(ffilt.real, ffilt.real, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # Take the second half of the autocorrelation
            autocorr_max = np.max(autocorr)  # Maximum value of autocorrelation
            autocorr_features.append(autocorr_max)

            # Analyze the signal envelope to find peaks
            analytical_signal = hilbert(ffilt.real)
            env = np.abs(analytical_signal)
            peaks, _ = find_peaks(env, distance=n)

            # Calculate distance based on peak position
            if len(peaks) > 0:
                pos_highest_peak = peaks[0]  # Position of the highest peak, assuming it's the first one
                distance = 0.5 * pos_highest_peak * dt * 2 * 343  # Calculate distance (assuming speed of sound 343 m/s)
                distances[i] = distance  # Store the distance for this signal
                peaks_list[i] = pos_highest_peak  # Store the peak position for this signal
                print(f"Row {i}: Distance = {distance} units, Peak Position = {pos_highest_peak}")
            else:
                distances[i] = np.nan  # Use NaN to indicate no peak/distance was detected
                peaks_list[i] = -1
                print(f"Row {i}: No peak detected.")
        peaks_list_all.append(peaks_list)

        # Combine features into a dataframe
        features = pd.DataFrame({
            'mean_freq': np.mean(filtered_signals, axis=1),
            'peak_freq': np.max(filtered_signals, axis=1),
            'variance_freq': np.var(filtered_signals, axis=1),
            'wavelet_energy': wavelet_features,
            'autocorr_max': autocorr_features,
            'peak_position': peaks_list,
            'distance': distances,
            'label': [label] * len(df.index)
        })

        all_features.append(features),peaks_list_all

    # Combine all feature dataframes into one
    combined_features = pd.concat(all_features, ignore_index=True)

    return combined_features, peaks_list_all

def group_labeled_data(peaks_list, signal_length, window_width):
    """
    Groups the labeled data based on peak locations and window width.

    Parameters:
    - peaks_list: List of peak positions for each signal.
    - signal_length: Length of the signals.
    - window_width: Width of the time window to group peaks.

    Returns:
    - y_label: A binary matrix indicating the presence of a peak in each time window.
    """
    # Calculate the number of windows in each signal
    n_windows = signal_length // window_width
    print("No of windows", n_windows)

    # Initialize the label matrix
    y_label = np.zeros((len(peaks_list), n_windows), dtype=int)

    # Loop through each signal and label the windows based on peak presence
    for i, peak in enumerate(peaks_list):
        if peak >= 0:  # Check if a peak was detected
            # Determine which window the peak falls into
            window_index = peak // window_width
            print("window number where peaks fall into", window_index)
            # Set the corresponding label to 1
            if window_index < n_windows:
                y_label[i, window_index] = 1
    return y_label

def save_to_npy(features_df, filename='processed_data.npy'):
    """
    Save the processed features DataFrame to a .npy file.
    """
    np.save(filename, features_df.to_numpy())
    print(f"Data saved to {filename}")
# Train Random Forest model

if __name__ == "__main__":
    start_time = time.time()

    # Load the dataset
    dataset_path = [
        '/kaggle/input/individual-project-test-dataset/test_adc_aluminiumfoil_1m.csv',
        '/kaggle/input/individual-project-test-dataset/test_adc_lessneedles_1m.csv',
        '/kaggle/input/individual-project-test-dataset/test_adc_lotofneedles_1m.csv',
        '/kaggle/input/individual-project-test-dataset/test_adc_plainwood_1m.csv',
        '/kaggle/input/individual-project-test-dataset/test_adc_steel_1m.csv'
    ]
    window_width = 64
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs

    # Corresponding labels for each file
    labels = ['aluminum', 'wood with lessneedles', 'wood with lotofneedles', 'plain wood', 'steel']
    data_list = read_and_prepare_data(dataset_path, labels)

    # Step 2: Extract Features from FFT Data
    features_df, peaks_list_all = extract_features(data_list, dt)

    # Check the shapes before proceeding
    print(f"Shape of features_df: {features_df.shape}")

    signal_length = data_list[0][0].shape[1]  # Take the length of the first signal as representative
    print(f"Corrected signal_length: {signal_length}")
    y_labels = [group_labeled_data(peaks_list, signal_length, window_width) for peaks_list in peaks_list_all]

    # Flatten y_labels to match the number of feature rows if necessary
    y_labels = np.concatenate(y_labels, axis=0)
    print(f"Shape of y_labels after flattening: {y_labels.shape}")

    # Ensure that the features and labels have matching numbers of samples
    if features_df.shape[0] != y_labels.shape[0]:
        raise ValueError(
            f"Mismatch in number of samples: features_df has {features_df.shape[0]} samples, y_labels has {y_labels.shape[0]} samples.")

    # Step 3: Save Processed Data to .npy File
    save_to_npy(features_df, 'processed_data.npy')
    np.save('labels.npy', y_labels)
    print(f"Labels saved to labels.npy")

    end_time = time.time()
    print("Time taken to preprocess the data:", end_time - start_time, "seconds")


