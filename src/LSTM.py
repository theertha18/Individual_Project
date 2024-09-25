import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras_tuner import RandomSearch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Utility Functions
def load_preprocessed_data(filepath, allow_pickle=True):
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


def print_evaluation_metrics(y_test, y_pred, model_name):
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    print(f"Confusion Matrix for {model_name}:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# LSTM Model with Hyperparameter Tuning
def build_lstm_model(hp, input_shape, num_classes):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32), return_sequences=True,
                        input_shape=input_shape))
    lstm_model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    lstm_model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32)))
    lstm_model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    lstm_model.add(Dense(units=num_classes, activation='softmax'))

    lstm_model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                       loss='categorical_crossentropy', metrics=['accuracy'])
    return lstm_model


def train_lstm_with_tuning(X_train, X_test, y_train, y_test, num_classes):
    input_shape = (X_train.shape[1], 1)
    tuner = RandomSearch(
        lambda hp: build_lstm_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='lstm_tuning',
        project_name='lstm_tune')

    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    best_lstm_model = tuner.get_best_models(num_models=1)[0]

    y_pred = np.argmax(best_lstm_model.predict(X_test), axis=1)
    print_evaluation_metrics(np.argmax(y_test, axis=1), y_pred, "LSTM (Tuned)")

    return best_lstm_model


# Main Function to Train LSTM Model
def train_and_evaluate_lstm(filepath, save_model=True, save_dir='./'):
    features_df = load_preprocessed_data(filepath)
    X = features_df.drop(columns=['label'])
    y = features_df['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    test_size_proportion = max(num_classes / len(y_encoded) + 0.05, 0.2)
    test_size_proportion = min(test_size_proportion, 0.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size_proportion, random_state=42,
                                                        stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for LSTM
    X_train_lstm = np.expand_dims(X_train_scaled, axis=2)
    X_test_lstm = np.expand_dims(X_test_scaled, axis=2)
    y_train_lstm = to_categorical(y_train, num_classes)
    y_test_lstm = to_categorical(y_test, num_classes)

    # Train LSTM Model with Hyperparameter Tuning
    lstm_model = train_lstm_with_tuning(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, num_classes)

    if save_model:
        lstm_model.save(f"{save_dir}lstm_model.h5")
        joblib.dump(scaler, f"{save_dir}scaler.pkl")
        print(f"Models saved to {save_dir}")

    return lstm_model, scaler


# Example usage:
train_and_evaluate_lstm('/kaggle/input/prepreocessednewdata/preprocessed_data (1).npy')
