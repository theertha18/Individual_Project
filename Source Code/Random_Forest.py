import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from skopt import BayesSearchCV  # For Bayesian Optimization
from sklearn.experimental import enable_halving_search_cv  # To enable HalvingGridSearchCV/Hyperband
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV  # Hyperband


# Load preprocessed data
def load_preprocessed_data(filepath, allow_pickle=True):
    """
    Loads preprocessed data from a .npy file and converts it to a pandas DataFrame.

    Parameters:
    - filepath: str, path to the .npy file containing the preprocessed data.
    - allow_pickle: bool, optional (default=True), whether to allow loading pickled object arrays.

    Returns:
    - pandas DataFrame containing features and labels.
    """
    data = np.load(filepath, allow_pickle=allow_pickle)
    columns = [f'feature_{i}' for i in range(data.shape[1] - 1)] + ['label']
    features_df = pd.DataFrame(data, columns=columns)
    return features_df


# Print evaluation metrics and confusion matrix
def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Prints the classification report and confusion matrix for a given model.

    Parameters:
    - y_test: array-like, true labels.
    - y_pred: array-like, predicted labels by the model.
    - model_name: str, the name of the model being evaluated.

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


# Plot learning curve to check for overfitting/underfitting
def plot_learning_curve(model, X, y, cv=5):
    """
    Plots learning curves showing the training and cross-validation accuracy over different training set sizes.

    Parameters:
    - model: sklearn estimator, the model to evaluate.
    - X: array-like, feature set.
    - y: array-like, label set.
    - cv: int, number of cross-validation folds (default=5).

    Returns:
    - None, but displays the learning curve plot.
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation Score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# Evaluate model with cross-validation
def evaluate_model_with_cross_validation(rf_model, X_train, y_train):
    """
    Evaluates a Random Forest model using cross-validation.

    Parameters:
    - rf_model: sklearn estimator, the trained Random Forest model.
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.

    Returns:
    - None, but prints the cross-validation scores and their mean.
    """
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores)}")


# Apply SMOTE for handling class imbalance
def apply_smote(X_train, y_train):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to the training data to handle class imbalance.

    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.

    Returns:
    - X_resampled: array-like, SMOTE-applied resampled training features.
    - y_resampled: array-like, SMOTE-applied resampled training labels.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# Recursive feature elimination (RFE)
def recursive_feature_elimination(X_train, y_train):
    """
    Performs Recursive Feature Elimination (RFE) to select the most important features.

    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.

    Returns:
    - selector: sklearn RFE model after fitting.
    """
    rf_model = RandomForestClassifier(random_state=42)
    selector = RFE(rf_model, n_features_to_select=10, step=1)
    selector = selector.fit(X_train, y_train)
    print(f"Selected Features: {selector.support_}")
    return selector


# Hyperparameter tuning with Grid Search
def tune_random_forest_with_grid_search(X_train, y_train):
    """
    Tunes hyperparameters for a Random Forest model using GridSearchCV.

    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.

    Returns:
    - best_estimator_: the Random Forest model with the best hyperparameters.
    """
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters (Grid Search): {grid_search.best_params_}")
    return grid_search.best_estimator_


# Hyperparameter tuning with Randomized Search
def tune_random_forest_with_random_search(X_train, y_train, n_iter=50):
    """
    Tunes hyperparameters for a Random Forest model using RandomizedSearchCV.

    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.
    - n_iter: int, number of parameter settings sampled (default=50).

    Returns:
    - best_estimator_: the Random Forest model with the best hyperparameters.
    """
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                       n_iter=n_iter, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='accuracy')
    random_search.fit(X_train, y_train)
    print(f"Best parameters (Randomized Search): {random_search.best_params_}")
    return random_search.best_estimator_


# Hyperparameter tuning with Bayesian Optimization (using skopt)
def tune_random_forest_with_bayesian_optimization(X_train, y_train):
    """
    Tunes hyperparameters for a Random Forest model using Bayesian Optimization (BayesSearchCV).

    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.

    Returns:
    - best_estimator_: the Random Forest model with the best hyperparameters found using Bayesian Optimization.
    """
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_dist = {
        'n_estimators': (50, 500),
        'max_depth': (10, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    bayes_search = BayesSearchCV(estimator=rf_model, search_spaces=param_dist, n_iter=32, cv=5, n_jobs=-1,
                                 random_state=42)
    bayes_search.fit(X_train, y_train)
    print(f"Best parameters (Bayesian Optimization): {bayes_search.best_params_}")
    return bayes_search.best_estimator_


# Hyperparameter tuning with Hyperband (using HalvingGridSearchCV and HalvingRandomSearchCV)
def tune_random_forest_with_hyperband(X_train, y_train, search_type="grid"):
    """
    Tunes hyperparameters for a Random Forest model using Hyperband via HalvingGridSearchCV or HalvingRandomSearchCV.

    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training label set.
    - search_type: str, either "grid" for HalvingGridSearchCV or "random" for HalvingRandomSearchCV.

    Returns:
    - best_estimator_: the Random Forest model with the best hyperparameters found using Hyperband.
    """
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [2, 4, 6],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    if search_type == "grid":
        hyperband_search = HalvingGridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2,
                                               scoring='accuracy')
    else:
        hyperband_search = HalvingRandomSearchCV(estimator=rf_model, param_distributions=param_grid, n_jobs=-1,
                                                 verbose=2, scoring='accuracy')

    hyperband_search.fit(X_train, y_train)
    print(f"Best parameters (Hyperband - {search_type.capitalize()} Search): {hyperband_search.best_params_}")
    return hyperband_search.best_estimator_


# Check for overfitting or underfitting
def check_overfitting_underfitting(rf_model, X_train, X_test, y_train, y_test):
    """
    Checks for overfitting or underfitting by comparing training and testing accuracy for the Random Forest model.

    Parameters:
    - rf_model: sklearn estimator, the trained Random Forest model.
    - X_train: array-like, training feature set.
    - X_test: array-like, test feature set.
    - y_train: array-like, training label set.
    - y_test: array-like, test label set.

    Returns:
    - None, but prints whether the model is overfitting, underfitting, or well-fitted.
    """
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = rf_model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    if train_accuracy > test_accuracy + 0.1:
        print("The model might be overfitting.")
    elif test_accuracy > train_accuracy + 0.1:
        print("The model might be underfitting.")
    else:
        print("The model is well-fitted.")


# Train and evaluate Random Forest with different hyperparameter tuning methods
def train_and_evaluate_rf(filepath, search_method="grid", n_iter=50, search_type="grid"):
    """
    Trains and evaluates a Random Forest model with different hyperparameter tuning methods (Grid Search, Randomized Search,
    Bayesian Optimization, or Hyperband).

    Parameters:
    - filepath: str, path to the preprocessed data file (.npy format).
    - search_method: str, method for hyperparameter tuning, one of ["grid", "random", "bayesian", "hyperband"].
    - n_iter: int, number of iterations for Randomized Search or Bayesian Optimization (default=50).
    - search_type: str, either "grid" or "random" for Hyperband (default="grid").

    Returns:
    - None, but prints evaluation metrics and tuning results, and plots the learning curve.
    """
    features_df = load_preprocessed_data(filepath)
    X = features_df.drop(columns=['label'])
    y = features_df['label']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Tune hyperparameters using different methods
    if search_method == "grid":
        rf_model = tune_random_forest_with_grid_search(X_train_smote, y_train_smote)
    elif search_method == "random":
        rf_model = tune_random_forest_with_random_search(X_train_smote, y_train_smote, n_iter=n_iter)
    elif search_method == "bayesian":
        rf_model = tune_random_forest_with_bayesian_optimization(X_train_smote, y_train_smote)
    elif search_method == "hyperband":
        rf_model = tune_random_forest_with_hyperband(X_train_smote, y_train_smote, search_type=search_type)
    else:
        raise ValueError("Invalid search_method. Must be one of ['grid', 'random', 'bayesian', 'hyperband']")

    # Evaluate the model for overfitting or underfitting
    check_overfitting_underfitting(rf_model, X_train, X_test, y_train, y_test)

    # Print evaluation metrics
    y_pred = rf_model.predict(X_test)
    print_evaluation_metrics(y_test, y_pred, f"Tuned Random Forest ({search_method.capitalize()} Search)")

    # Plot learning curve
    plot_learning_curve(rf_model, X_train, y_train)

    # Perform cross-validation
    evaluate_model_with_cross_validation(rf_model, X_train, y_train)


# Example usage of Random Forest with Grid Search
train_and_evaluate_rf('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy', search_method="grid")

# Example usage of Random Forest with Randomized Search
train_and_evaluate_rf('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy',
                      search_method="random", n_iter=50)

# Example usage of Random Forest with Bayesian Optimization
train_and_evaluate_rf('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy',
                      search_method="bayesian")

# Example usage of Random Forest with Hyperband
train_and_evaluate_rf('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy',
                      search_method="hyperband", search_type="grid")

train_and_evaluate_rf('/kaggle/input/individual-project-preprocessed-data/preprocessed_data.npy',
                      search_method="hyperband")
