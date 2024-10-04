import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import randint


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


# MLP Model with Hyperparameter Tuning
def train_mlp_model_with_randomized_search(X_train, X_test, y_train, y_test):
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (100, 100), (50, 50)],
        'activation': ['logistic', 'relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }

    mlp_model = MLPClassifier(max_iter=500, random_state=42)
    random_search = RandomizedSearchCV(mlp_model, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1,
                                       random_state=42, verbose=1)
    random_search.fit(X_train, y_train)

    best_mlp_model = random_search.best_estimator_
    print(f"\nBest Hyperparameters for MLP: {random_search.best_params_}")

    y_pred = best_mlp_model.predict(X_test)
    print_evaluation_metrics(y_test, y_pred, "Multi-Layer Perceptron (Randomized Search)")

    return best_mlp_model


# Main Function to Train MLP Model
def train_and_evaluate_mlp(filepath, save_model=True, save_dir='./'):
    features_df = load_preprocessed_data(filepath)
    X = features_df.drop(columns=['label'])
    y = features_df['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    test_size_proportion = max(len(np.unique(y_encoded)) / len(y_encoded) + 0.05, 0.2)
    test_size_proportion = min(test_size_proportion, 0.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size_proportion, random_state=42,
                                                        stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP Model with Hyperparameter Tuning
    mlp_model = train_mlp_model_with_randomized_search(X_train_scaled, X_test_scaled, y_train, y_test)

    if save_model:
        joblib.dump(mlp_model, f"{save_dir}mlp_model.pkl")
        joblib.dump(scaler, f"{save_dir}scaler.pkl")
        print(f"Models saved to {save_dir}")

    return mlp_model, scaler


# Example usage:
train_and_evaluate_mlp('/kaggle/input/prepreocessednewdata/preprocessed_data (1).npy')
