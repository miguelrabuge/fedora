import warnings
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report

from fedora.core.engine import Fedora
from fedora.core.utilities.lib import split_data, get_features

class CustomFedora(Fedora):
    # Addings new attributes (max_features)
    def __init__(self, max_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_features = max_features                                # Maximum allowed features in a phenotype

    # Adding another component to the fitness function
    def evaluate(self, phenotype):
        fitness, info = super().evaluate(phenotype)
        penalty = len(get_features(phenotype)) / self.max_features      # Penalize individuals that have a higher number of features
        fitness = fitness + penalty                                     # SGE minimizes, so adding positive values becomes a penalty
        return fitness, info

    # Adding bests' number of features to progress_report.csv
    def evolution_progress(self, population):
        best = sorted(population, key=lambda x: x['fitness'])[0]        # Selecting the individual with best fitness
        return "%4d" % (len(get_features(best['phenotype'])))           # Adding its number of features it to progress_report.csv
    
    # Stopping if there is no fitness improvement after 5 generations
    def stop_criteria(self, it, best, population):
        best_fitness, best_generation = best                            # Load the fitness and generation of the best individual
        if it == best_generation + 5:                                   # Stop if the best is the same for the past 5 generations
            warnings.warn("Terminated by stopping criteria")            # Log it in warnings.txt file
            return True
        return False
    
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
    custom_fedora = CustomFedora(
        max_features=12,
        seed = SEED,
        model = DecisionTreeClassifier(),
        error_metric = errorBAcc,
        sge_parameters_path = "custom-mnist.yml",
        grammar_path = "custom-mnist.pybnf",
        logging_dir = "./"
    )
    dt = DecisionTreeClassifier(random_state=SEED)
    sc = StandardScaler()

    # Pipelines
    pipeline_baseline = Pipeline([
        ("classifier", dt)
    ])

    pipeline_custom_fedora = Pipeline([
        ("feature-engineering", custom_fedora),
        ("scaler", sc),                  
        ("classifier", dt)
    ])

    # Fitting and Testing
    baseline_pred = pipeline_baseline.fit(**train).predict(test['X'])
    fedora_pred = pipeline_custom_fedora.fit(**train).predict(test['X'])

    for predictions in [baseline_pred, fedora_pred]:
        print(classification_report(test['y'], predictions, digits=4))