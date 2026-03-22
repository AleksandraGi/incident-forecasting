import json
import numpy as np
import pandas as pd


def is_in_incident(timestamp, windows):
    """
    Checks whether a given timestamp belongs to any incident window

    Parameters:
        timestamp: a single timestamp from the dataframe
        windows: list of incident intervals (start, end)
    Returns:
        1: timestamp is inside any incident window
        0: timestamp is NOT inside any incident window
    """
    # Check whether a given timestamp belongs to any labeled incident window
    for start, end in windows:
        # If the timestamp is inside one of the intervals, return 1
        if start <= timestamp <= end:
            return 1
    return 0


def create_sliding_windows(values, incidents, window_size=12, horizon=6):
    """
    Creates input-output samples using a sliding window

    Input:
        previous window_size values
    Label:
        1: an incident occurs within the next horizon steps
        0: otherwise
    Parameters:
        values: numpy array with metric values
        inciodents: numpy array with 0/1 incident labels for each timestamp
        window_size: number of past points used as input
        horizon: number of future points checked for an incident
    Returns:
        X: numpy array of shape (n_samples, window_size)
        Y: numpy array of shape (n_samples, )
    """
    X = []
    y = []

    # Total number of points in the time series
    n = len(values)

    # Move the end of the window through the series
    # Start from the first index where a full input window exists
    # Stop early enough so that a full future horizon still exists
    for end_idx in range(window_size - 1, n - horizon):
        # Calculate the start index of the current input window
        start_idx = end_idx - window_size + 1

        # Extract the past window used as model input
        x_window = values[start_idx:end_idx + 1]

        # Extract the future incident labels for the prediction horizon
        future_incidents = incidents[end_idx + 1:end_idx + 1 + horizon]

        # Extract the future incident labels for the prediction horizon. otherwise assing 0
        label = 1 if np.max(future_incidents) > 0 else 0

        # Store the input window and its label
        X.append(x_window)
        y.append(label)

    # Convert lists into numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

def prepare_dataset(
    csv_path="data/raw/machine_temperature_system_failure.csv",
    labels_path="data/raw/combined_windows.json",
    series_key="realKnownCause/machine_temperature_system_failure.csv",
    window_size=12,
    horizon=6,     
):
    """
    Loads raw NAB data and converts it

    1. Load time series CSV
    2. Load anomaly windows from JSON
    3. Create incident column
    4. Create sliding windows

    Retrns:
        df: original dataframe with additional "incident" column
        X: model inputs
        y: labels
    """

    # Laod raw CSV file into a dataframe
    df = pd.read_csv(csv_path)
    # Convert timestamp column from text into datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Sort rows by time and reset indexing
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Open JSON file with anomaly windows and load content as python dictionaries
    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Get anomaly windows only for the selected time series
    anomaly_windows = labels[series_key]
    # Convert the window boundries from text into datetime format
    anomaly_windows = [ (pd.to_datetime(start), pd.to_datetime(end)) for start, end in anomaly_windows ]

    # name every timestamp and check if is in incidents 
    # (creates table:       timestamp   value   incident)  
    #                       ...         71.0    0
    # 1 if the timestamp belongs to any anomaly window and 0 otherwise
    df["incident"] = df["timestamp"].apply(lambda timestamp: is_in_incident(timestamp, anomaly_windows))

    # Extract the metric values as a numpy array
    values = df["value"].values
    # Extract the binary incident labels as a numpy array
    incidents = df["incident"].values

    # Convert raw time series into learning samples
    X, y = create_sliding_windows(
        values=values,
        incidents=incidents,
        window_size=window_size,
        horizon=horizon,
    )

    return df, X, y