import subprocess
import pandas as pd
import pytest


def test_dataframe_shapes(prepare_stage):
    df_train, df_test = prepare_stage

    assert df_train.shape == (60000, 785)
    assert df_test.shape == (10000, 785)


@pytest.fixture(name="prepare_stage", scope="module")
def fixture_prepare_stage():
    subprocess.run(["dvc", "repro", "prepare"], check=False)

    df_train = pd.read_csv("/workspace/data/interim/mnist_train.csv", header=None)
    df_test = pd.read_csv("/workspace/data/interim/mnist_test.csv", header=None)

    return (df_train, df_test)
