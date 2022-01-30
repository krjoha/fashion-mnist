import os
import numpy as np
import yaml
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


TRAIN_FILE = "data/processed/mnist_train.npz"
OUTPUT_FILE = "models/model.joblib"

os.makedirs("models", exist_ok=True)
params = yaml.safe_load(open("params.yaml"))

train = np.load(TRAIN_FILE)
train_data = train["data"]
train_labels = train["labels"]
train_labels_binary = label_binarize(train_labels, classes=[*range(10)])

# If need for validation set
# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#    train_data,
#    train_labels_binary,
#    test_size=params["train"]["val_split"],
#    random_state=params["seed"],
# )

model = OneVsRestClassifier(
    LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=params["train"]["num_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=params["seed"]
    )
)

model.fit(train_data, train_labels_binary)
dump(model, OUTPUT_FILE)
