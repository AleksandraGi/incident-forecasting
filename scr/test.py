# file names in combined_windows.json

import json

labels_path = "data/raw/combined_windows.json"
with open(labels_path, "r") as f:
    labels = json.load(f)

# print(labels.keys())

for key in labels.keys():
    print(key)