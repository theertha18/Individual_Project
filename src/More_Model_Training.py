import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM


def load_preprocessed_data(filepath, allow_pickle=True):
    """
    Loads preprocessed data from a .npy file and converts it into a pandas DataFrame.
    """
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Prints the classification report and confusion matrix for a given model's predictions.
    """
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    print(f"Confusion Matrix for {model_name}:")
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_feature_importance(model, X, feature_names):
    """
    Plots the feature importance for tree-based models like Random Forest.
    """
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Random Forest")
    plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()


def plot_logistic_regression_coefficients(model, X, feature_names):
    """
    Plots the absolute value of coefficients for a Logistic Regression model.
    """
    coefficients = np.abs(model.coef_[0])
    sorted_indices = np.argsort(coefficients)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Coefficients - Logistic Regression")
    plt.bar(range(X.shape[1]), coefficients[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()


def train_svm(X_train, X_test, y_train, y_test):
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print_evaluation_metrics(y_test, y_pred, "Support Vector Machine")

    return svm_model


def train_mlp(X_train, X_test, y_train, y_test):
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)

    print_evaluation_metrics(y_test, y_pred, "Multilayer Perceptron")

    return mlp_model


def train_cnn(X_train, X_test, y_train, y_test):
    # Reshape data to 3D for Conv1D (samples, time steps, features)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    cnn_model = Sequential([
        tf.keras.Input(shape=(X_train_cnn.shape[1], 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

    # Predict on the test set
    y_pred = cnn_model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Evaluate the model
    print_evaluation_metrics(y_test, y_pred_classes, "Convolutional Neural Network (CNN)")
    loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test)
    print(f"CNN Test Accuracy: {accuracy}")

    return cnn_model, accuracy


def train_lstm(X_train, X_test, y_train, y_test):
    # Reshape data to 3D for LSTM (samples, time steps, features)
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    lstm_model = Sequential([
        tf.keras.Input(shape=(X_train_lstm.shape[1], 1)),
        LSTM(100),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

    # Predict on the test set
    y_pred = lstm_model.predict(X_test_lstm)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Evaluate the model
    print_evaluation_metrics(y_test, y_pred_classes, "Long Short-Term Memory (LSTM)")
    loss, accuracy = lstm_model.evaluate(X_test_lstm, y_test)
    print(f"LSTM Test Accuracy: {accuracy}")

    return lstm_model, accuracy


def train_and_evaluate_models(filepath, save_models=True, save_dir='/kaggle/working/'):
    """
    Loads preprocessed data, splits it into training and testing sets, scales the features,
    and trains and evaluates multiple machine learning models.
    """
    # Load preprocessed data from the specified file
    features_df = load_preprocessed_data(filepath)

    # Separate features and labels
    X = features_df.drop(columns=['label'])  # Feature matrix
    y = features_df['label']  # Labels

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Train and evaluate traditional models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'KNN Classifier': KNeighborsClassifier(),
        'Random Forest Classifier': RandomForestClassifier(),
    }

    # Dictionary to store model scores
    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Predict on the test set
        model_scores[name] = model.score(X_test, y_test)
        print_evaluation_metrics(y_test, y_pred, name)

        if name == 'Random Forest Classifier':
            plot_feature_importance(model, X_train, X.columns)

        if name == 'Logistic Regression':
            plot_logistic_regression_coefficients(model, X_train, X.columns)

        if save_models:
            model_filename = f"{save_dir}{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")

    # Train and evaluate SVM, MLP, CNN, and LSTM
    svm_model = train_svm(X_train, X_test, y_train, y_test)
    mlp_model = train_mlp(X_train, X_test, y_train, y_test)
    cnn_model, cnn_accuracy = train_cnn(X_train, X_test, y_train, y_test)
    lstm_model, lstm_accuracy = train_lstm(X_train, X_test, y_train, y_test)

    # Add their scores to the dictionary
    model_scores['Support Vector Machine'] = svm_model.score(X_test, y_test)
    model_scores['Multilayer Perceptron'] = mlp_model.score(X_test, y_test)
    model_scores['CNN'] = cnn_accuracy
    model_scores['LSTM'] = lstm_accuracy

    # Save the CNN and LSTM models
    if save_models:
        cnn_model.save(f"{save_dir}cnn_model.h5")
        lstm_model.save(f"{save_dir}lstm_model.h5")
        print("CNN and LSTM models saved.")

    # Print all model scores for comparison
    print("Model Scores:")
    print(model_scores)


# Call the function to train and evaluate models
train_and_evaluate_models('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy')
