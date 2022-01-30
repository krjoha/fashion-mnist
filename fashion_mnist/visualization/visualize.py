import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = "data/interim/mnist_test.csv"
OUTPUT_PATH = "reports/figures/"

os.makedirs(OUTPUT_PATH, exist_ok=True)
df = pd.read_csv(INPUT_PATH, header=None)

img_per_label = 30
img_size = 28

samples = df.groupby([0]).head(img_per_label).sort_values([0])
samples = samples.drop(columns=[0])
samples = samples.to_numpy()

image = np.ones((img_size * 10, img_size * img_per_label))

for i in range(img_per_label):
    for j in range(img_per_label):
        offset = i * img_per_label + j
        if offset < img_per_label * 10:
            image[
                i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size
            ] = np.reshape(samples[offset], (img_size, img_size))

plt.figure(figsize=(15, 15))
plt.axis("off")
plt.imsave(os.path.join(OUTPUT_PATH, "zalando-mnist-sprite.png"), image, cmap="gray")
