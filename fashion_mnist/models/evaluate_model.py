import csv
import json
import math
import numpy as np
import os
from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer

MODEL_FILE = "models/model.joblib"
TEST_FILE = "data/processed/mnist_test.npz"
OUTPUT_PATH = "models/"
LABELS = {
    0: "t_shirt_top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle_boots",
}

os.makedirs(OUTPUT_PATH, exist_ok=True)

model = load(MODEL_FILE)
test = np.load(TEST_FILE)
test_data = test["data"]
test_labels = test["labels"]
num_labels = len(LABELS)
label_binarizer = LabelBinarizer()
test_labels = label_binarizer.fit_transform(test_labels)

predictions = model.predict(test_data)
prediction_probabilities = model.predict_proba(test_data)

accuracy = accuracy_score(test_labels, predictions)
precision, recall, fscore, _ = precision_recall_fscore_support(
    test_labels, predictions, average="weighted"
)

precision_curves = {}
recall_curves = {}
prc_thresholds = {}
for i in range(num_labels):
    precision_curves[i], recall_curves[i], prc_thresholds[i] = precision_recall_curve(
        test_labels[:, i], prediction_probabilities[:, i]
    )

fpr = {}
tpr = {}
roc_thresholds = {}
roc_auc = {}
for i in range(num_labels):
    fpr[i], tpr[i], roc_thresholds[i] = roc_curve(
        test_labels[:, i], prediction_probabilities[:, i]
    )
    roc_auc[i] = auc(fpr[i], tpr[i])

with open(os.path.join(OUTPUT_PATH, "metrics.json"), "w") as fd:
    json.dump(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": fscore,
            "roc_auc": sum(roc_auc.values()) / len(roc_auc.values()),
        },
        fd,
        indent=4,
    )


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


for i in range(num_labels):
    nth_point = math.ceil(len(prc_thresholds[i]) / 1000)
    prc_points = list(zip(precision_curves[i], recall_curves[i], prc_thresholds[i]))[
        ::nth_point
    ]
    with open(os.path.join(OUTPUT_PATH, f"prc-{i}.json"), "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
            cls=NpEncoder,
        )

    with open(os.path.join(OUTPUT_PATH, f"roc-{i}.json"), "w") as fd:
        json.dump(
            {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr[i], tpr[i], roc_thresholds[i])
                ]
            },
            fd,
            indent=4,
            cls=NpEncoder,
        )

with open(os.path.join(OUTPUT_PATH, "predictions.csv"), "w") as fd:

    writer = csv.writer(fd)
    writer.writerow(["label", "prediction"])

    test_labels_unbin = label_binarizer.inverse_transform(test_labels)
    predictions_unbin = label_binarizer.inverse_transform(predictions)

    for label, prediction in zip(test_labels_unbin, predictions_unbin):
        label = LABELS[label]
        prediction = LABELS[prediction]
        writer.writerow([label, prediction])
