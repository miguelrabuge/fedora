import pandas as pd

from fedora.core.engine import Fedora
from fedora.core.utilities.lib import get_features
from fedora.sge.utilities.protected_math import Infix

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def errorAcc(true, pred):
    return 1 - accuracy_score(true, pred)


def MyOpt(series: pd.Series):
    return series.apply(lambda x: x.count("i"))

def concatLen(s1: pd.Series, s2: pd.Series):
    return pd.Series(s1 + s2).apply(lambda x: len(x))


if __name__ == "__main__":
    SEED = 105

    features = pd.DataFrame({
        "feature0": pd.Series(["Test i", "Test ii", "Test iii"]),
        "feature1": pd.Series(["Hello", "World", "!"])
    })
    labels = [0, 0, 1]

    # Pipeline Components
    fedora = Fedora(
        seed = SEED,
        model = DecisionTreeClassifier(),
        error_metric = errorAcc,
        sge_parameters_path = "strings.yml",
        grammar_path = "strings.pybnf",
        logging_dir = "./",
        operators = {"_MyOpt_": Infix(MyOpt), "_cclen_": Infix(concatLen)}
    )
    new_features = fedora.fit_transform(features, labels)
    print(new_features, end="\n\n")

    best = Fedora.get_best("strings-results/", SEED)
    features = get_features(best["phenotype"])

    for i, feature in enumerate(features):
        print(f"feature_{i}: {feature}")

