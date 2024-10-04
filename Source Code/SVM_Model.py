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
    """
    Loads preprocessed data from a .npy file and converts it into a pandas DataFrame.

    Parameters:
    - filepath: str, path to the .npy file containing the data.
    - allow_pickle: bool, optional (default=True), whether to allow loading pickled object arrays.

    Returns:
    - features_df: pandas DataFrame containing features and labels.
    """
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Prints and logs the classification report and confusion matrix for a given model.

    Parameters:
    - y_test: array-like, true labels.
    - y_pred: array-like, predicted labels from the model.
    - model_name: str, the name of the model being evaluated.

    Returns:
    - None, but displays confusion matrix heatmap and prints metrics.
    """
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


def apply_smote(X_train, y_train):
    """
    Applies SMOTE to the training dataset to handle class imbalance by oversampling the minority classes.

    Parameters:
    - X_train: array-like, training features.
    - y_train: array-like, training labels.

    Returns:
    - X_train_res: array-like, resampled training features.
    - y_train_res: array-like, resampled training labels.
    """
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


def train_svm(X_train, X_test, y_train, y_test, class_weight=None):
    """
    Trains an SVM model on the training data and evaluates it on the test data. It also checks for overfitting or underfitting.

    Parameters:
    - X_train: array-like, training features.
    - X_test: array-like, test features.
    - y_train: array-like, training labels.
    - y_test: array-like, test labels.
    - class_weight: dict, optional, weights associated with classes in the form {class_label: weight}.

    Returns:
    - svm_model: trained SVM model.
    """
    svm_model = SVC(kernel='rbf', class_weight=class_weight, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print_evaluation_metrics(y_test, y_pred, "Support Vector Machine")

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


def plot_learning_curves(estimator, X, y):
    """
    Plots learning curves for a given estimator, showing training and validation accuracy as the training set size increases.

    Parameters:
    - estimator: sklearn estimator, the model to evaluate.
    - X: array-like, feature set.
    - y: array-like, label set.

    Returns:
    - None, but displays the learning curves plot.
    """
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


def tune_svm_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning on an SVM model using GridSearchCV.

    Parameters:
    - X_train: array-like, training features.
    - y_train: array-like, training labels.

    Returns:
    - best_estimator_: the SVM model with the best combination of hyperparameters.
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3, 4, 5]
    }

    print("\n--- Performing GridSearchCV ---")
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    grid_search = GridSearchCV(SVC(), param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters from Grid Search: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy from Grid Search: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def random_search_svm(X_train, y_train):
    """
    Performs hyperparameter tuning on an SVM model using RandomizedSearchCV.

    Parameters:
    - X_train: array-like, training features.
    - y_train: array-like, training labels.

    Returns:
    - best_estimator_: the SVM model with the best combination of hyperparameters found by random search.
    """
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


def evaluate_with_cross_validation(svm_model, X_train, y_train):
    """
    Evaluates the performance of the SVM model using cross-validation.

    Parameters:
    - svm_model: the trained SVM model to evaluate.
    - X_train: array-like, training features.
    - y_train: array-like, training labels.

    Returns:
    - None, but prints the cross-validation scores and mean accuracy.
    """
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")


def check_overfitting_optimized_model(svm_model, X_train, X_test, y_train, y_test):
    """
    Checks for overfitting or underfitting by comparing training and testing accuracy for an optimized model.

    Parameters:
    - svm_model: the trained SVM model to evaluate.
    - X_train: array-like, training features.
    - X_test: array-like, test features.
    - y_train: array-like, training labels.
    - y_test: array-like, test labels.

    Returns:
    - None, but prints a message about overfitting or underfitting.
    """
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


def save_model(model, filename):
    """
    Saves the trained model to a file using joblib.

    Parameters:
    - model: the trained model to save.
    - filename: str, the file path where the model will be saved.

    Returns:
    - None, but saves the model to the specified file.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def train_and_evaluate_svm(filepath):
    """
    Main function that trains and evaluates an SVM model pipeline, including data loading, preprocessing, model training,
    evaluation, hyperparameter tuning, and saving the model.

    Parameters:
    - filepath: str, path to the preprocessed data file (.npy format).

    Returns:
    - None, but performs and prints model evaluation, learning curve plots, and saves optimized models.
    """
    features_df = load_preprocessed_data(filepath)

    X = features_df.drop(columns=['label'])
    y = features_df['label']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 2.0, 4: 1.0}
    svm_model = train_svm(X_train_smote, X_test, y_train_smote, y_test, class_weight=class_weights)

    plot_learning_curves(svm_model, X_train_smote, y_train_smote)

    evaluate_with_cross_validation(svm_model, X_train_smote, y_train_smote)

    svm_model_grid = tune_svm_hyperparameters(X_train_smote, y_train_smote)

    y_pred_grid = svm_model_grid.predict(X_test)
    print_evaluation_metrics(y_test, y_pred_grid, "Optimized SVM (Grid Search)")

    check_overfitting_optimized_model(svm_model_grid, X_train_smote, X_test, y_train_smote, y_test)

    save_model(svm_model_grid, "optimized_svm_grid.pkl")

    svm_model_random = random_search_svm(X_train_smote, y_train_smote)

    y_pred_random = svm_model_random.predict(X_test)
    print_evaluation_metrics(y_test, y_pred_random, "Optimized SVM (Random Search)")

    check_overfitting_optimized_model(svm_model_random, X_train_smote, X_test, y_train_smote, y_test)

    save_model(svm_model_random, "optimized_svm_random.pkl")


# Call the main function to train and evaluate SVM
train_and_evaluate_svm('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy')
