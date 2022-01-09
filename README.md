fashion-mnist
==============================

Using the fashion-mnist dataset to demonstrate ML-pipelines and experimentation with DVC.


To build a docker image:
```
docker build -t fashion-mnist --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile .
```

To start a docker container, expose GPU, mount current directory and connect to the container:
```
docker run -d --rm -it --gpus all --volume $(pwd):/workspace --name fashion-mnist fashion-mnist
docker exec -it fashion-mnist /bin/bash
```

If you want to use python environments instead, use the commands below. That creates a virtual python environment in your current folder, activates it and installs dependencies with pip
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Initialize ```git``` and ```dvc``` and create an initial commit before doing anything else:
```
git init
dvc init
git add .
git commit -m "Initial commit"
```

Use ```doit``` to run linting and tests:
```
doit lint
doit pytest
```

Use ```dvc``` to create reproducible ML-pipelines and experiments with git tracking:
```
dvc repro
dvc exp run
```

Use ```mlflow``` to run lots of experiments that you do not want to track with git.


Project Organization
------------

    ├── LICENSE
    ├── dodo.py            <- Makefile-like multiplatform CLI
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources (ex. script config files)
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentation
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── fashion_mnist                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes fashion_mnist a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    |
    ├── Dockerfile         <- Dockerfile with settings to run scripts in Docker container
    ├── dvc.yaml           <- DVC pipeline; see dvc.org
    ├── params.yaml        <- Parameter values (things like hyperparameters) used by DVC pipeline
    ├── setup.cfg          <- config file with settings for running pylint, flake8 and bandit
    └── pytest.ini         <- config file with settings for running pytest


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience.</small>