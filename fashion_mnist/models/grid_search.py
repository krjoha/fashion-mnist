import itertools
import subprocess

# Automated grid search experiments
max_depth_values = [10, 20, 30, 40]
num_estimators_values = [100, 200, 300, 400]

# Iterate over all combinations of hyperparameter values.
for max_depth, num_estimators in itertools.product(
    max_depth_values, num_estimators_values
):
    # Execute "dvc exp run --queue --set-param train.max_depth=<max_depth>
    #   --set-param train.num_estimators=<num_estimators>".
    subprocess.run(
        [
            "dvc",
            "exp",
            "run",
            "--queue",
            "--set-param",
            f"train.max_depth={max_depth}",
            "--set-param",
            f"train.num_estimators={num_estimators}",
        ]
    )

subprocess.run(["dvc", "exp", "run", "--run-all"])
