import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Function to load features and labels (replace with your actual data loading logic)
def load_data(features_path='processed_data.npy', labels_path='labels.npy'):
    features = np.load(features_path)
    labels = np.load(labels_path)

    # Flatten the labels if they are windowed
    labels = labels.reshape(labels.shape[0], -1)

    # Adjust features to match the flattened labels
    features = features[:, :-2]  # Exclude the last columns for labels matching

    return features, labels


# Function to evaluate the model
def evaluate_model(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n" + "=" * 50 + "\n")


# This block will only run if the script is executed directly (not imported)
if __name__ == "__main__":
    # Load the data
    X, Y = load_data()  # Replace with your actual data loading function

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define the models
    np.random.seed(42)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'KNN Classifier': KNeighborsClassifier(),
        'Random Forest Classifier': RandomForestClassifier()
    }

    # Train the models and evaluate them
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        model_scores[name] = model.score(X_test, Y_test)
        evaluate_model(model, X_test, Y_test, name)

    # Optionally print the model scores
    print("Model Scores:")
    print(model_scores)
