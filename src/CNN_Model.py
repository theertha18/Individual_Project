import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib  # For saving models


# Custom Keras wrapper to work like KerasClassifier
class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom wrapper for Keras models to work like KerasClassifier for easier integration with scikit-learn functions.

    Parameters:
    - build_fn: function to build the Keras model.
    - epochs: int, number of epochs to train the model.
    - batch_size: int, batch size for training.
    - verbose: int, verbosity level for model training.
    - fit_args: additional arguments for fitting the model.

    Methods:
    - fit: Fits the model to the training data.
    - predict: Makes predictions using the trained model.
    - score: Evaluates the model accuracy on given data.
    """

    def __init__(self, build_fn, epochs=10, batch_size=32, verbose=0, **fit_args):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.fit_args = fit_args
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = self.build_fn()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **self.fit_args)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


# Function to load preprocessed data
def load_preprocessed_data(filepath, allow_pickle=True):
    """
    Loads preprocessed data from a .npy file and converts it to a pandas DataFrame.

    Parameters:
    - filepath: str, path to the .npy file containing the preprocessed data.
    - allow_pickle: bool, whether to allow pickled object arrays (default=True).

    Returns:
    - pandas DataFrame containing features and labels.
    """
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


# Hypermodel function for Keras Tuner
def build_model(hp):
    """
    Builds a Sequential Keras model with hyperparameters dynamically set by Keras Tuner.

    Parameters:
    - hp: keras_tuner.HyperParameters object, used to define the hyperparameters for the model.

    Returns:
    - model: a compiled Keras model.
    """
    model = Sequential()
    input_length = X_train.shape[1]  # Input sequence length
    max_kernel_size = min(input_length, 5)  # Restrict kernel size based on input length

    model.add(Input(shape=(input_length, 1)))  # Define input shape for Conv1D

    # Conv1D layer with dynamically set hyperparameters
    model.add(Conv1D(
        filters=hp.Choice('filters', values=[32, 64, 128]),
        kernel_size=hp.Choice('kernel_size', values=[2, 3, 4, max_kernel_size]),
        activation=hp.Choice('activation', values=['relu', 'tanh']),
        padding='same'
    ))

    # MaxPooling layer
    model.add(MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2, 3])))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice('dense_units', values=[64, 128, 256]),
        activation=hp.Choice('activation', values=['relu', 'tanh']))
    )

    # Output layer for multi-class classification
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Function to print evaluation metrics and plot confusion matrix
def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Prints the classification report and confusion matrix for a model's predictions.

    Parameters:
    - y_test: array-like, true labels.
    - y_pred: array-like, predicted labels by the model.
    - model_name: str, name of the model being evaluated.

    Returns:
    - None, but prints metrics and displays confusion matrix heatmap.
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


# Load and prepare data
def prepare_data(filepath):
    """
    Loads, encodes, and splits data into training and test sets, followed by scaling.

    Parameters:
    - filepath: str, path to the preprocessed data file (.npy format).

    Returns:
    - X_train, X_test: numpy arrays, scaled feature matrices for training and test sets.
    - y_train, y_test: numpy arrays, labels for training and test sets.
    """
    features_df = load_preprocessed_data(filepath)

    X = features_df.drop(columns=['label'])  # Feature matrix
    y = features_df['label']  # Labels

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for Conv1D input (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test


# Plot learning curves
def plot_learning_curves(estimator, X_train, y_train):
    """
    Plots learning curves showing the training and validation accuracy as the training size increases.

    Parameters:
    - estimator: sklearn-compatible estimator, model to evaluate.
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.

    Returns:
    - None, but displays the learning curves plot.
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=5, scoring='accuracy',
                                                            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
                                                            random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, label="Validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Training set size")
    plt.title("Learning Curves (CNN)")
    plt.legend()
    plt.grid()
    plt.show()


# Check for overfitting or underfitting
def check_overfitting_underfitting(model, X_train, X_test, y_train, y_test):
    """
    Checks for overfitting or underfitting by comparing training and test accuracy.

    Parameters:
    - model: trained Keras model.
    - X_train, X_test: numpy arrays, feature matrices for training and test sets.
    - y_train, y_test: numpy arrays, labels for training and test sets.

    Returns:
    - None, but prints whether the model is overfitting, underfitting, or well-fitted.
    """
    train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]  # Training accuracy
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]  # Testing accuracy

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    if train_accuracy > test_accuracy + 0.1:
        print("The model might be overfitting.\n")
    elif test_accuracy > train_accuracy + 0.1:
        print("The model might be underfitting.\n")
    else:
        print("The model is likely well-fitted.\n")


# Hyperparameter tuning using Keras Tuner
def perform_hyperparameter_search(X_train, X_test, y_train, y_test):
    """
    Performs hyperparameter tuning using Keras Tuner's RandomSearch and trains the best model.

    Parameters:
    - X_train, X_test: numpy arrays, feature matrices for training and test sets.
    - y_train, y_test: numpy arrays, labels for training and test sets.

    Returns:
    - best_model: the Keras model with the best hyperparameters.
    """
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='cnn_tuning'
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of filters is {best_hps.get('filters')}, 
    kernel size is {best_hps.get('kernel_size')}, and activation function is {best_hps.get('activation')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    """)

    # Rebuild the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

    # Check for overfitting or underfitting
    check_overfitting_underfitting(best_model, X_train, X_test, y_train, y_test)

    # Generate predictions on the test set
    y_pred = np.argmax(best_model.predict(X_test), axis=1)

    # Print evaluation metrics (classification report and confusion matrix)
    print_evaluation_metrics(y_test, y_pred, "CNN")

    # Save the model
    best_model.save('best_cnn_model.h5')
    print("Model saved as '/kaggle/working/best_cnn_model.h5'")

    # Plot learning curves
    wrapped_model = KerasClassifier(build_fn=lambda: best_model, epochs=20, batch_size=32, verbose=0)
    plot_learning_curves(wrapped_model, X_train, y_train)

    return best_model


# Main pipeline execution
filepath = '/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy'
X_train, X_test, y_train, y_test = prepare_data(filepath)
best_model = perform_hyperparameter_search(X_train, X_test, y_train, y_test)
