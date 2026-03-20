from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from preprocess import prepare_dataset


def time_based_train_test_split(X, y, test_size=0.2):
    """
    Splits the dataset into train and test parts without shuffling
    Earlier samples go to train, later samples go to test
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
    Fits a Logistic Regression model on training data.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    df, X, y = prepare_dataset(window_size=12, horizon=6)

    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_logistic_regression(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nMetrics on test set:")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred))