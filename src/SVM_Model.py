import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, \
    RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import logging
import joblib

# Set up logging to file
logging.basicConfig(filename='svm_training.log', level=logging.INFO)


# Utility functions
def load_preprocessed_data(filepath, allow_pickle=True):
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


def print_evaluation_metrics(y_test, y_pred, model_name):
    report = classification_report(y_test, y_pred)
    logging.info(f"--- Classification Report for {model_name} ---\n{report}")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    logging.info(f"--- Confusion Matrix for {model_name} ---\n{cm}")


# SMOTE function for handling class imbalance
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


# SVM Training function with class weighting
def train_svm(X_train, X_test, y_train, y_test, class_weight=None):
    # Custom class weights can be passed as argument
    svm_model = SVC(kernel='rbf', class_weight=class_weight, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Evaluate the SVM model
    print_evaluation_metrics(y_test, y_pred, "Support Vector Machine")

    # Check for overfitting or underfitting
    train_accuracy = svm_model.score(X_train, y_train)
    test_accuracy = svm_model.score(X_test, y_test)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    if train_accuracy > test_accuracy + 0.1:
        print("The model might be overfitting.\n")
    elif test_accuracy > train_accuracy + 0.1:
        print("The model might be underfitting.\n")
    else:
        print("The model is likely well-fitted.\n")

    return svm_model


# Learning curve plotting function
def plot_learning_curves(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, label="Validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Training set size")
    plt.title("Learning Curves (SVM)")
    plt.legend()
    plt.grid()
    plt.show()


# Hyperparameter tuning with GridSearchCV
def tune_svm_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],  # Expanded C range
        'gamma': [0.001, 0.01, 0.1, 1, 'scale'],  # Wider gamma search
        'kernel': ['rbf', 'linear', 'poly'],  # Various kernels
        'degree': [2, 3, 4, 5]  # More degree values for poly kernel
    }

    print("\n--- Performing GridSearchCV ---")
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    grid_search = GridSearchCV(SVC(), param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters from Grid Search: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy from Grid Search: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# RandomizedSearchCV - an alternative
def random_search_svm(X_train, y_train):
    param_dist = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3, 4, 5]
    }

    print("\n--- Performing RandomizedSearchCV ---")
    random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1,
                                       random_state=42)
    random_search.fit(X_train, y_train)

    print(f"\nBest parameters from Random Search: {random_search.best_params_}")
    print(f"Best cross-validation accuracy from Random Search: {random_search.best_score_:.4f}")
    return random_search.best_estimator_


# Cross-validation function
def evaluate_with_cross_validation(svm_model, X_train, y_train):
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")


# Function to check for overfitting/underfitting after optimization
def check_overfitting_optimized_model(svm_model, X_train, X_test, y_train, y_test):
    train_accuracy = svm_model.score(X_train, y_train)
    test_accuracy = svm_model.score(X_test, y_test)

    print(f"Optimized Model - Training Accuracy: {train_accuracy:.4f}")
    print(f"Optimized Model - Testing Accuracy: {test_accuracy:.4f}")

    if train_accuracy > test_accuracy + 0.1:
        print("The optimized model might be overfitting.\n")
    elif test_accuracy > train_accuracy + 0.1:
        print("The optimized model might be underfitting.\n")
    else:
        print("The optimized model is likely well-fitted.\n")


# Saving model to a file
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


# Main function to train and evaluate SVM
def train_and_evaluate_svm(filepath):
    # Load preprocessed data
    features_df = load_preprocessed_data(filepath)

    # Separate features and labels
    X = features_df.drop(columns=['label'])
    y = features_df['label']

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE to handle class imbalance
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Train the SVM model with class weighting
    class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 2.0, 4: 1.0}  # Example of custom class weights
    svm_model = train_svm(X_train_smote, X_test, y_train_smote, y_test, class_weight=class_weights)

    # Plot learning curves
    plot_learning_curves(svm_model, X_train_smote, y_train_smote)

    # Evaluate with cross-validation
    evaluate_with_cross_validation(svm_model, X_train_smote, y_train_smote)

    # --- Grid Search ---
    svm_model_grid = tune_svm_hyperparameters(X_train_smote, y_train_smote)

    # Evaluate the optimized model from Grid Search
    y_pred_grid = svm_model_grid.predict(X_test)
    print_evaluation_metrics(y_test, y_pred_grid, "Optimized SVM (Grid Search)")

    # Check for overfitting/underfitting for the optimized model from Grid Search
    check_overfitting_optimized_model(svm_model_grid, X_train_smote, X_test, y_train_smote, y_test)

    # Save the optimized model from Grid Search
    save_model(svm_model_grid, "optimized_svm_grid.pkl")

    # --- Randomized Search ---
    svm_model_random = random_search_svm(X_train_smote, y_train_smote)

    # Evaluate the optimized model from Randomized Search
    y_pred_random = svm_model_random.predict(X_test)
    print_evaluation_metrics(y_test, y_pred_random, "Optimized SVM (Random Search)")

    # Check for overfitting/underfitting for the optimized model from Randomized Search
    check_overfitting_optimized_model(svm_model_random, X_train_smote, X_test, y_train_smote, y_test)

    # Save the optimized model from Randomized Search
    save_model(svm_model_random, "optimized_svm_random.pkl")


# Call the main function to train and evaluate SVM
train_and_evaluate_svm('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy')
