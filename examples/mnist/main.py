import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report


from fedora.core.engine import Fedora
from fedora.core.utilities.lib import split_data

def errorBAcc(true, pred):
    return 1 - balanced_accuracy_score(true, pred)

if __name__ == "__main__":
    SEED = 1

    # Loading Dataset
    mnist = load_digits()
    
    features = pd.DataFrame(mnist.data, columns=[f"c{i}" for i in range(64)])
    labels = pd.DataFrame(mnist.target)

    # Splitting dataset
    train, test = split_data(features, labels, SEED, test_size=0.2)

    # Pipeline Components
    fedora = Fedora(
        seed = SEED,
        model = DecisionTreeClassifier(),
        error_metric = errorBAcc,
        sge_parameters_path = "mnist.yml",
        grammar_path = "mnist.pybnf",
        logging_dir = "./"
    )
    dt = DecisionTreeClassifier(random_state=SEED)
    sc = StandardScaler()

    # Pipelines
    pipeline_baseline = Pipeline([
        ("classifier", dt)
    ])

    pipeline_fedora = Pipeline([
        ("feature-engineering", fedora),
        ("scaler", sc),                  
        ("classifier", dt)
    ])

    # Fitting and Testing
    baseline_pred = pipeline_baseline.fit(**train).predict(test['X'])
    fedora_pred = pipeline_fedora.fit(**train).predict(test['X'])

    for predictions in [baseline_pred, fedora_pred]:
        print(classification_report(test['y'], predictions, digits=4))