import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



def load_preprocessed_data(filepath, allow_pickle=True):
    """
    Loads preprocessed data from a .npy file and converts it into a pandas DataFrame.

    Parameters:
    - filepath: str, path to the .npy file containing the preprocessed data.
    - allow_pickle: bool, whether to allow loading of pickled objects in numpy (default is True).

    Returns:
    - features_df: DataFrame containing the loaded data with columns for features and labels.
    """
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Prints the classification report and confusion matrix for a given model's predictions.

    Parameters:
    - y_test: Array-like, true labels of the test set.
    - y_pred: Array-like, predicted labels by the model.
    - model_name: str, name of the model being evaluated.

    Returns:
    - None. Displays the classification report and confusion matrix as a heatmap.
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

    Parameters:
    - model: Trained model, expected to have a feature_importances_ attribute.
    - X: Feature matrix used for training.
    - feature_names: List of strings, the names of the features.

    Returns:
    - None. Displays a bar plot of feature importances.
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

    Parameters:
    - model: Trained Logistic Regression model.
    - X: Feature matrix used for training.
    - feature_names: List of strings, the names of the features.

    Returns:
    - None. Displays a bar plot of the absolute coefficients.
    """
    coefficients = np.abs(model.coef_[0])
    sorted_indices = np.argsort(coefficients)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Coefficients - Logistic Regression")
    plt.bar(range(X.shape[1]), coefficients[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()


def train_and_evaluate_models(filepath, save_models=True, save_dir='../Models/'):
    """
    Loads preprocessed data, splits it into training and testing sets, scales the features,
    and trains and evaluates multiple machine learning models.

    Models trained:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Random Forest

    For each model, it prints evaluation metrics and plots relevant feature analysis.

    Parameters:
    - filepath: str, path to the .npy file containing the preprocessed data.
    - save_models: bool, if True, saves the trained models to disk (default is True).
    - save_dir: str, directory where models will be saved (default is '../Saved_Models/').

    Returns:
    - None. Displays evaluation metrics, confusion matrices, and feature importance plots.
    """
    # Load preprocessed data from the specified file
    features_df = load_preprocessed_data(filepath)

    # Separate features and labels
    X = features_df.drop(columns=['label'])  # Feature matrix
    y = features_df['label']  # Labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define the models in a dictionary
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'KNN Classifier': KNeighborsClassifier(),
        'Random Forest Classifier': RandomForestClassifier()
    }

    # Dictionary to store model scores
    model_scores = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Predict on the test set

        # Store the score of the model
        model_scores[name] = model.score(X_test, y_test)

        # Evaluate and print metrics
        print_evaluation_metrics(y_test, y_pred, name)

        # Feature importance for Random Forest
        if name == 'Random Forest Classifier':
            plot_feature_importance(model, X_train, X.columns)

        # Coefficients for Logistic Regression
        if name == 'Logistic Regression':
            plot_logistic_regression_coefficients(model, X_train, X.columns)

        # Save the model if requested
        if save_models:
            model_filename = f"{save_dir}{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")

    # Print all model scores for comparison
    print("Model Scores:")
    print(model_scores)



# Call the function to train and evaluate models
train_and_evaluate_models('../Dataset/Processed/preprocessed_data.npy')
