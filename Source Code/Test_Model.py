import numpy as np
import pandas as pd
import joblib
import os
from scipy.signal import get_window, correlate, find_peaks
from tensorflow.keras.models import load_model

# Manual mapping of numeric labels to actual class names
label_mapping = {0: 'aluminum', 1: 'plain wood', 2: 'steel', 3: 'wood with lessneedles', 4: 'wood with lotofneedles'}


# Function to load the CSV file and prepare the data for prediction
def load_and_prepare_test_data(file_path, dt):
    """
    Loads and preprocesses the CSV file for testing.

    Parameters:
    - file_path: str, path to the CSV file.
    - dt: float, time interval between samples (1 / sampling frequency).

    Returns:
    - test_features: numpy array containing the processed features ready for model prediction.
    """
    # Load the dataset (assuming no header)
    dataframe = pd.read_csv(file_path, header=None)

    # Select columns starting from the 17th (index 16) onwards
    df = dataframe.iloc[:, 16:]

    # Preprocessing steps (same as used in training)
    sinad_list = []
    peak_counts = []
    peaks_list = []
    autocorr_features = []

    low_freq = 30000  # 30 kHz
    high_freq = 50000  # 50 kHz

    for i in range(len(df.index)):
        f = df.iloc[i, :]

        # Apply window function to the signal
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

        # Calculate SINAD
        signal_power = np.sum(np.abs(fhat_filtered) ** 2)
        noise_power = np.sum(np.abs(fhat[~range_mask]) ** 2)
        sinad = 10 * np.log10(signal_power / noise_power)
        sinad_list.append(sinad)

        # Detect and count peaks
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

    # Combine the extracted features
    test_features = pd.DataFrame({
        'sinad': sinad_list,
        'peak_count': peak_counts,
        'peak_position': peaks_list,
        'autocorr_max': autocorr_features
    })

    # Rename columns to match the trained model's expected feature names
    feature_columns = [f'feature_{i}' for i in range(test_features.shape[1])]
    test_features.columns = feature_columns

    return test_features


def apply_window(signal, window_type='hann'):
    """
    Applies a windowing function to a signal to reduce spectral leakage.

    Parameters:
    - signal: ndarray, the original signal to be windowed.
    - window_type: str or tuple, specifying the type of window to apply.

    Returns:
    - The windowed signal as an array.
    """
    window = get_window(window_type, len(signal))
    return signal * window


def predict_with_model(model_path, test_data_path, scaler_path, sampling_frequency):
    """
    Loads the model based on the file extension and makes predictions on the test data.
    Supports both scikit-learn (.pkl) and Keras (.h5) models.

    Parameters:
    - model_path: str, path to the model file (.pkl or .h5).
    - test_data_path: str, path to the CSV file containing test data.
    - scaler_path: str, path to the saved scaler for data normalization.
    - sampling_frequency: float, the sampling frequency used in data processing.

    Returns:
    - y_pred_labels: Predicted labels (actual names) by the model.
    """
    # Load test data
    dt = 1 / sampling_frequency
    test_features = load_and_prepare_test_data(test_data_path, dt)

    # Check if the model is a scikit-learn model (.pkl) or Keras model (.h5)
    model_extension = os.path.splitext(model_path)[1]

    # Load the scaler (common for both model types)
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(test_features)

    # Predict based on the model type
    if model_extension == '.pkl':
        # Scikit-learn model
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
    elif model_extension == '.h5':
        # Keras model
        model = load_model(model_path)
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)  # Reshape for CNN/LSTM
        y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    else:
        raise ValueError("Unsupported model file type. Please use '.pkl' for scikit-learn or '.h5' for Keras models.")

    # Map numeric predictions to actual labels using the manual mapping
    y_pred_labels = [label_mapping[label] for label in y_pred]

    return y_pred_labels


# Example usage:
sampling_frequency = 1953125  # Sampling frequency used in data processing
test_data_path = '/kaggle/input/individual-project-dataset/adc_plainwood_1m.csv'  # Path to the test CSV file

# The model path will be manually changed based on which model is being tested
model_path = '/kaggle/input/tuned-models/tuned_rf_random_model.pkl.h5'  # Can change to KNN SVM, Random Forest, MLP, CNN, or LSTM
scaler_path = '/kaggle/input/tuned-models/scaler.pkl'

# Run the model prediction
predictions = predict_with_model(model_path, test_data_path, scaler_path, sampling_frequency)
print("Predictions (Label Names):", predictions)
