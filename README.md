# incident-forecasting

I use the NAB dataset file realKnownCause/machine_temperature_system_failure.csv. Each training sample is built with a sliding window: the input contains the previous W temperature values, and the label is 1 if an incident occurs within the next H time steps.

data source: 
https://github.com/numenta/NAB
csv_path: https://github.com/numenta/NAB/blob/master/data/realKnownCause/machine_temperature_system_failure.csv
labels_path: https://github.com/numenta/NAB/blob/master/labels/combined_windows.json
series_key: https://github.com/numenta/NAB/tree/master/data/realKnownCause
