import gzip
import os
import requests

DATA_RAW_PATH = os.path.join("data", "raw")
DATA_INTERIM_PATH = os.path.join("data", "interim")

os.makedirs(DATA_RAW_PATH, exist_ok=True)
os.makedirs(DATA_INTERIM_PATH, exist_ok=True)

train_images = "train-images-idx3-ubyte.gz"
train_labels = "train-labels-idx1-ubyte.gz"
test_images = "t10k-images-idx3-ubyte.gz"
test_labels = "t10k-labels-idx1-ubyte.gz"

filenames = [train_images, train_labels, test_images, test_labels]

for filename in filenames:
    with open(os.path.join(DATA_RAW_PATH, filename), "wb") as f:
        r = requests.get(
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" + filename
        )
        f.write(r.content)


def convert(imgfile, labelfile, outfile, num):

    f = gzip.open(os.path.join(DATA_RAW_PATH, imgfile), "rb")
    o = open(os.path.join(DATA_INTERIM_PATH, outfile), "w")
    l = gzip.open(os.path.join(DATA_RAW_PATH, labelfile), "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(num):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


convert(train_images, train_labels, "mnist_train.csv", 60000)
convert(test_images, test_labels, "mnist_test.csv", 10000)
