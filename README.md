# incident-forecasting

I use the NAB dataset file realKnownCause/machine_temperature_system_failure.csv. Each training sample is built with a sliding window: the input contains the previous W temperature values, and the label is 1 if an incident occurs within the next H time steps.

data source: 
https://github.com/numenta/NAB
csv_path: https://github.com/numenta/NAB/blob/master/data/realKnownCause/machine_temperature_system_failure.csv
labels_path: https://github.com/numenta/NAB/blob/master/labels/combined_windows.json
series_key: https://github.com/numenta/NAB/tree/master/data/realKnownCause


I evaluated the model across multiple alert thresholds. As expected, lower thresholds increased recall but also produced more false alarms, while higher thresholds improved alert precision at the cost of missing more incidents. In this experiment, a threshold of 0.3 provided the best balance, achieving precision 0.927, recall 0.977, and F1-score 0.952.

The confusion matrix helps interpret the alerting behavior more directly. At a threshold of 0.5, the model generated no false positives, which means every alert was correct, but it missed some real incidents. At a threshold of 0.3, the model detected more incidents, at the cost of introducing a small number of false alarms.

The confusion matrices make the threshold trade-off very clear.
With a threshold of 0.3, the model detected 559 incidents and missed only 13, but produced 44 false positives.
With a threshold of 0.5, the model produced no false positives at all, but missed 64 real incidents and detected 508.
This shows the classic alerting trade-off: a lower threshold improves sensitivity, while a higher threshold improves alert precision.




NA CZYSTO

# Incident Forecasting from Time-Series Metrics
## Project Overview

This project implements a binary classification model that predicts whether an incident will occur within the next **H** time steps based on the previous **W** steps of a time-series metric.

The task is formulated using a **sliding-window approach**:
- the input contains the previous `W` observations,
- the target label is `1` if an incident occurs within the next `H` time steps, otherwise `0`.

The goal is not to detect whether the current point is anomalous, but to **forecast an upcoming incident** in the near future.

---

## Dataset

This project uses a public dataset from the **Numenta Anomaly Benchmark (NAB)**:
- file: `realKnownCause/machine_temperature_system_failure.csv`

This time series represents the temperature of an internal component of a large industrial machine. The dataset also includes labeled anomalous intervals, which are treated here as incident periods.

Labels were taken from:
- `labels/combined_windows.json`

In this project:
- `incident = 1` means that the timestamp belongs to one of the labeled anomaly windows,
- `incident = 0` otherwise.

---

## Problem Formulation

The original time series is converted into a supervised learning dataset.

For each time step:
- the model receives the previous `W` values as input,
- the label is `1` if an incident appears in the following `H` time steps.

### Example

If:
- `W = 12`
- `H = 6`

then each sample is built as:
- **input**: the previous 12 values of the metric,
- **label**: `1` if at least one incident occurs in the next 6 time steps, otherwise `0`.

This formulation makes the task closer to a real alerting scenario, where the objective is to raise an alert before the incident fully happens.

---

## Preprocessing

The preprocessing pipeline performs the following steps:

1. Load the raw CSV time series.
2. Load anomaly intervals from the NAB labels JSON file.
3. Create a binary `incident` column for each timestamp.
4. Build sliding-window samples.
5. Split the dataset chronologically into train and test sets.

A chronological split is used instead of a random split in order to better reflect real forecasting conditions and avoid leakage from the future.

---

## Model

As a baseline, I used **Logistic Regression** from scikit-learn.

This choice was made because:
- it is simple and interpretable,
- it provides a strong baseline for binary classification,
- it is easy to compare later with more advanced sequence models.

Before training, the input windows are scaled with `StandardScaler`, fitted only on the training set.

---

## Evaluation Setup

The model is evaluated on the test set using:
- **Precision**
- **Recall**
- **F1-score**

These metrics are more informative than accuracy for this task because the incident class is less frequent than the non-incident class.

In addition, I evaluated the model under multiple **alert thresholds** using predicted probabilities:
- 0.1
- 0.3
- 0.5
- 0.7
- 0.9

This is important because in a real alerting system the threshold controls the trade-off between:
- catching more incidents,
- and avoiding false alarms.

---

## Results

### Dataset statistics
- Total samples: **22,678**
- Positive labels: **2,288**
- Negative labels: **20,390**

### Threshold comparison

| Threshold | Precision | Recall | F1-score |
|----------:|----------:|-------:|---------:|
| 0.1 | 0.8301 | 0.9825 | 0.8999 |
| 0.3 | 0.9270 | 0.9773 | 0.9515 |
| 0.5 | 1.0000 | 0.8881 | 0.9407 |
| 0.7 | 1.0000 | 0.7098 | 0.8303 |
| 0.9 | 1.0000 | 0.0280 | 0.0544 |

The best overall trade-off was achieved at **threshold = 0.3**, which gave both high precision and very high recall.

A threshold of **0.5** was more conservative: it produced no false positives, but missed more incidents.

---

## Confusion Matrix Analysis

### Threshold = 0.3
- True Negatives (TN): **3920**
- False Positives (FP): **44**
- False Negatives (FN): **13**
- True Positives (TP): **559**

At this threshold, the model detects almost all incidents and misses only a small number of them, but it introduces a small number of false alarms.

### Threshold = 0.5
- True Negatives (TN): **3964**
- False Positives (FP): **0**
- False Negatives (FN): **64**
- True Positives (TP): **508**

At this threshold, every alert is correct, but more real incidents are missed.

This illustrates a standard alerting trade-off:
- a lower threshold improves sensitivity,
- a higher threshold improves alert precision.

---

## Interpretation

The baseline Logistic Regression model performed very well on this selected NAB series.

The results suggest that the temperature signal contains a detectable pre-incident pattern. In particular:
- lower thresholds are useful when incident coverage is more important,
- higher thresholds are useful when false alarms are expensive.

For this dataset, **threshold = 0.3** appears to be the best practical balance.

---

## Limitations

This project has several limitations:

1. Only a single time series was used.
2. Sliding-window samples overlap heavily, so neighboring samples are strongly correlated.
3. The evaluation is point-wise, while real alerting systems are often evaluated at the event level.
4. Logistic Regression is a simple baseline and may not capture more complex temporal dependencies.
5. The anomaly labels from NAB are treated directly as incident labels, which is a simplification.

---

## Possible Improvements

Possible next steps include:
- testing additional NAB series,
- combining multiple metrics,
- adding validation split and hyperparameter tuning,
- comparing with sequence models such as 1D CNN, LSTM, or GRU,
- evaluating event-level alert quality,
- adding visualizations of the time series, predictions, and thresholds.

---

## Project Structure

```text
incident-forecasting/
├── data/
│   └── raw/
│       ├── machine_temperature_system_failure.csv
│       └── combined_windows.json
├── main.py
├── src/
    ├── preprocess.py
    ├── test.py
│   └── train.py
├── README.md
└── requirements.txt
```

## Conclusion
This project demonstrates a simple but effective incident forecasting pipeline based on a sliding-window formulation of a time-series anomaly dataset.

Even a simple Logistic Regression baseline was able to achieve strong results on the selected NAB series. The threshold analysis showed how model behavior changes depending on the desired balance between sensitivity and alert precision, which is an important consideration in real-world alerting systems.