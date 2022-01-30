import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/interim/"
OUTPUT_PATH = "data/processed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

df_train = pd.read_csv(os.path.join(INPUT_PATH, "mnist_train.csv"), header=None)
df_test = pd.read_csv(os.path.join(INPUT_PATH, "mnist_test.csv"), header=None)

train_data = df_train.drop(columns=[0]).to_numpy()
train_labels = df_train[0].to_numpy()
test_data = df_test.drop(columns=[0]).to_numpy()
test_labels = df_test[0].to_numpy()

scaler = StandardScaler().fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

np.savez_compressed(
    os.path.join(OUTPUT_PATH, "mnist_train.npz"), data=train_data, labels=train_labels
)
np.savez_compressed(
    os.path.join(OUTPUT_PATH, "mnist_test.npz"), data=test_data, labels=test_labels
)
