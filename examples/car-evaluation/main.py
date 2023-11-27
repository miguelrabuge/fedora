import ssl
import pandas as pd

from ucimlrepo import fetch_ucirepo 

from fedora.core.engine import Fedora
from fedora.core.utilities.lib import split_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report


def errorBAcc(true, pred):
    return 1 - balanced_accuracy_score(true, pred)

if __name__ == "__main__":
    SEED = 42

    # Load Dataset
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = fetch_ucirepo(id=19) 

    # Extract Features
    features = dataset.data.features
    labels = dataset.data.targets["class"]
    labels, names = pd.factorize(labels)
    

    # Split data
    train_val, test = split_data(features, labels, SEED, test_size=0.2)

    # Pipeline Components
    fedora = Fedora(
        seed = SEED,
        model = DecisionTreeClassifier(),
        error_metric = errorBAcc,
        sge_parameters_path = "car-evaluation.yml",
        grammar_path = "car-evaluation.pybnf",
        logging_dir = "./"
    )
    dt = DecisionTreeClassifier(random_state=SEED)
    oe = OneHotEncoder(dtype=bool, sparse_output=False).set_output(transform="pandas")

    # Pipelines
    pipeline_baseline = Pipeline([
        ("encoder", oe),
        ("classifier", dt)
    ])

    pipeline_fedora = Pipeline([
        ("encoder", oe),
        ("feature-engineering", fedora),
        ("classifier", dt)
    ])

    # Fitting and Scoring
    baseline_pred = pipeline_baseline.fit(**train_val).predict(test['X'])
    fedora_pred = pipeline_fedora.fit(**train_val).predict(test['X'])

    for predictions in [baseline_pred, fedora_pred]:
        print(classification_report(test['y'], predictions, target_names=names, digits=4))

        


