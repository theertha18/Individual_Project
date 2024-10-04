import numpy as np
import pandas as pd
from scipy.signal import get_window, find_peaks
import time
from scipy.signal import correlate


def read_and_prepare_data(dataset_path, labels):
    """
    Reads multiple CSV files and prepares them by selecting specific columns.

    Parameters:
    - dataset_path: List of file paths to the datasets.
    - labels: List of labels corresponding to each dataset.

    Returns:
    - data_list: A list of tuples containing the prepared DataFrame and its corresponding label.
    """
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
    """
    Extracts features from each signal in the datasets, including SINAD, peak count,
    peak position, and autocorrelation maximum.

    Parameters:
    - data_list: List of tuples, where each tuple contains a DataFrame and a label.
    - dt: The time interval between samples, based on the sampling frequency.

    Returns:
    - combined_features: A DataFrame containing the extracted features for all datasets.
    """
    all_features = []

    # Constants for frequency range selection
    low_freq = 30000  # 30 kHz
    high_freq = 50000  # 50 kHz

    for df, label in data_list:
        # Initialize lists to store features for each signal in the dataset
        sinad_list = []
        peak_counts = []
        peaks_list = []
        autocorr_features = []

        for i in range(len(df.index)):
            f = df.iloc[i, :]

            # Apply window function to the signal to reduce edge effects
            windowed_signal = apply_window(f)

            # Fourier Transform for frequency analysis
            n = len(windowed_signal)
            fhat = np.fft.fft(windowed_signal, n)
            freqs = np.fft.fftfreq(n, dt)
            PSD = fhat * np.conj(fhat) / n

            # Select frequency range between 30 and 50 kHz
            range_mask = (freqs >= low_freq) & (freqs <= high_freq)
            fhat_filtered = fhat[range_mask]
            PSD_filtered = PSD[range_mask]

            # Calculate SINAD (Signal-to-Noise and Distortion Ratio)
            signal_power = np.sum(np.abs(fhat_filtered) ** 2)
            noise_power = np.sum(np.abs(fhat[~range_mask]) ** 2)
            sinad = 10 * np.log10(signal_power / noise_power)
            sinad_list.append(sinad)

            # Detect and count peaks in the selected frequency range
            peaks, _ = find_peaks(np.abs(fhat_filtered))
            peak_count = len(peaks)
            peak_counts.append(peak_count)

            # Store the first peak position or -1 if no peak found
            peaks_list.append(peaks[0] if len(peaks) > 0 else -1)

            # Autocorrelation to find periodic components in the signal
            autocorr = correlate(windowed_signal, windowed_signal, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # Take the second half of the autocorrelation
            autocorr_max = np.max(autocorr)
            autocorr_features.append(autocorr_max)

        # Combine extracted features into a DataFrame for the current dataset
        features = pd.DataFrame({
            'sinad': sinad_list,
            'peak_count': peak_counts,
            'peak_position': peaks_list,
            'autocorr_max': autocorr_features,
            'label': [label] * len(df.index)
        })

        # Append the features of the current dataset to the list of all features
        all_features.append(features)

    # Combine all feature dataframes into one DataFrame
    combined_features = pd.concat(all_features, ignore_index=True)

    # Print the extracted features for verification
    print("Extracted Features:")
    print(combined_features)

    return combined_features


def save_to_npy(features_df, filename='processed_data.npy'):
    """
    Save the processed features DataFrame to a .npy file.

    Parameters:
    - features_df: DataFrame containing the features to save.
    - filename: The filename for the saved .npy file.
    """
    np.save(filename, features_df.to_numpy())
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    start_time = time.time()

    # Define the dataset paths
    dataset_path = [
        r'D:\individualProject\Dataset\adc_aluminiumfoil_1m.csv',
        r'D:\individualProject\Dataset\adc_lessneedles_1m.csv',
        r'D:\individualProject\Dataset\adc_lotofneedles_1m.csv',
        r'D:\individualProject\Dataset\adc_plainwood_1m.csv',
        r'D:\individualProject\Dataset\adc_steel_1m.csv'
    ]
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs  # Time interval between samples

    # Corresponding labels for each file
    labels = ['aluminum', 'wood with lessneedles', 'wood with lotofneedles', 'plain wood', 'steel']

    # Read and prepare data
    data_list = read_and_prepare_data(dataset_path, labels)

    # Extract features from the data
    features_df = extract_features(data_list, dt)

    # Save the processed features to a .npy file
    save_to_npy(features_df, '../Dataset/Processed/preprocessed_data.npy')

    end_time = time.time()
    print("Time taken to preprocess the data:", end_time - start_time, "seconds")
