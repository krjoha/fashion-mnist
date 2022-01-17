import subprocess
import numpy as np
import pytest


class Datasets:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels


def test_dataset_shapes(featurize_stage):

    assert np.shape(featurize_stage.train_data) == (60000, 784)
    assert np.shape(featurize_stage.train_labels) == (60000,)
    assert np.shape(featurize_stage.test_data) == (10000, 784)
    assert np.shape(featurize_stage.test_labels) == (10000,)


def test_dataset_standardized(featurize_stage):

    train_mean = np.mean(featurize_stage.train_data)
    train_std = np.std(featurize_stage.train_data)

    np.testing.assert_almost_equal(train_mean, 0)
    np.testing.assert_almost_equal(train_std, 1)


@pytest.fixture(name="featurize_stage", scope="module")
def fixture_featurize_stage():
    subprocess.run(["dvc", "repro", "featurize"], check=False)

    train = np.load("/workspace/data/processed/mnist_train.npz")
    test = np.load("/workspace/data/processed/mnist_test.npz")

    return Datasets(train["data"], train["labels"], test["data"], test["labels"])
