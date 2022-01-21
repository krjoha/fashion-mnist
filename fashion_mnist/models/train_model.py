import json
import os
import numpy as np
import sys
import yaml
from joblib import dump
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open("params.yaml"))
data_path = sys.argv[1]
model_file = sys.argv[2]
scores_file = sys.argv[3]
os.makedirs(data_path, exist_ok=True)

train = np.load(os.path.join(data_path, "mnist_train.npz"))
train_data = train["data"]
train_labels = train["labels"]


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    train_data,
    train_labels,
    test_size=params["train"]["val_split"],
    random_state=params["seed"],
)

model = RandomForestClassifier(
    n_estimators=params["train"]["num_estimators"],
    n_jobs=-1,
    max_depth=params["train"]["max_depth"],
    random_state=params["seed"],
)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

dump(model, model_file)
with open(scores_file, "w") as f:
    json.dump({"avg_acc": accuracy}, f, indent=4)
