from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from preprocess import prepare_dataset

import numpy as np

def time_based_train_test_split(X, y, test_size=0.2):
    """
    Splits the dataset into train and test parts without shuffling
    Earlier samples - train
    Later samples - test
    """
    # amount of all samples
    n_samples = len(X)

    # example: if test_size = 0.2:
        # 80% is train set
        # 20% is test set
    split_index = int(n_samples * (1 - test_size))

    # spliting all samples into train and test sets
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Fits a Logistic Regression model on training data
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def apply_threshold(probabilites, threshold=0.5):
    """
    Convert predictad probabilities into binary predictions
    """
    return (probabilites >= threshold).astype(int)

if __name__ == "__main__":
    # Prepare dataset
    df, X, y = prepare_dataset(window_size=12, horizon=6)

    print("Total samples:", len(X))
    print("Positive labels:", y.sum())
    print("Negative labels:", len(y) - y.sum())

    # Time-based split
    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=0.2)

    print("Train shape: ", X_train.shape)
    print("Test shape: ", X_test.shape)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = train_logistic_regression(X_train_scaled, y_train)

    # Predicted probabilities for class 1
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nProbability diagnostics:")
    print("Min probability:", y_proba.min())
    print("Max probability:", y_proba.max())
    print("First 20 probabilities:", y_proba[:20])

    y_pred_direct = model.predict(X_test_scaled)
    y_pred_threshold = apply_threshold(y_proba, threshold=0.5)

    print("\nComparison of predictions:")
    print("Direct predict unique values:", np.unique(y_pred_direct, return_counts=True))
    print("Threshold predict unique values:", np.unique(y_pred_threshold, return_counts=True))
    print("Are they identical?", np.array_equal(y_pred_direct, y_pred_threshold))

    # Different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]


    for threshold in thresholds:
        y_pred = apply_threshold(y_proba, threshold=threshold)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\nThreshold: {threshold}")
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-score:", round(f1, 4))

    # Report for the threshold 0.5
    y_pred_05 = apply_threshold(y_proba, threshold=0.5)

    print("\nDetailed classification report for threshold = 0.5:")
    print(classification_report(y_test, y_pred_05, zero_division=0))
