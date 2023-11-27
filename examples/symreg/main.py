""" A Symbolic Regression Example using the SGE Algorithm """

import numpy as np

from fedora.sge import EngineSGE

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def quartic_polynomial(x):
    return pow(x, 4) + pow(x, 3) + pow(x, 2) + x


class SymbolicRegression(EngineSGE):
    def __init__(self, sge_parameters_file, function):
        super().__init__(sge_parameters_file)
        self.function = function
        self.train_data, self.test_data = self.get_data()
        self.best = [float("inf"), "phenotype"]

    def get_data(self, n_vars=1, start=-5, end=5.4, step=0.1, test_size=0.7):
        # Generate Data
        X = [np.arange(start, end, step) for var in range(n_vars)]    # [[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]]
        y = np.array(list(map(self.function, *X)))
        X = list(zip(*X))                                             # [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return {"X": X_train, "y": y_train}, {"X": X_test, "y": y_test}


    def evaluate(self, phenotype : str) -> tuple[float, dict]:
        """ Phenotype example: ((x[0]+x[1])-1.0) """

        def error(predicted, label, error_metric=mean_squared_error):
            return error_metric(label, predicted)

        predictions = [eval(phenotype) for x in self.train_data["X"]]    
        labels = self.train_data["y"]

        fitness = error(predictions, labels)

        if fitness < self.best[0]:
            self.best = [fitness, phenotype]

        return fitness, None

    def stop_criteria(self, it, best, population):
        return best[0] == 0

if __name__ == "__main__":
    symreg = SymbolicRegression("symreg.yml", quartic_polynomial)
    symreg.evolutionary_algorithm()
    
    print(f"Best Fitness: {symreg.best[0]}")
    print(f"Best Phenotype: {symreg.best[1]}")

