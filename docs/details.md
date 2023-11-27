# Further Details

## Analysing the Best individuals

If we look at the car-evaluation's /run\_42/best.json file:

```json
{"features": 21, "phenotype": "~x['safety_med'],x['buying_high'],~x['safety_low'],~x['doors_2'],x['lug_boot_big'],~x['persons_2'],~x['persons_4'],x['maint_low'],~x['buying_med'],x['lug_boot_small'],x['doors_2'],x['buying_high'],x['maint_high'],(~x['safety_high']&x['lug_boot_small']),~x['maint_high'],~x['maint_low'],~x['buying_vhigh'],x['lug_boot_small'],(x['maint_vhigh']&~x['maint_high']),x['safety_med'],~x['persons_more']&(x['persons_4']&x['doors_3'])"}
```

We are able to see that the transformation has 21 features. It is possible to individually get each feature in a list by doing:

```python
from fedora.core.utilities.lib import get_features

individual = Fedora.get_best("car-evaluation-results/", 42)
features = get_features(individual["phenotype"])

print(features)
```

```python
["~x['safety_med']", "x['buying_high']", "~x['safety_low']", "~x['doors_2']", "x['lug_boot_big']", "~x['persons_2']", "~x['persons_4']", "x['maint_low']", "~x['buying_med']", "x['lug_boot_small']", "x['doors_2']", "x['buying_high']", "x['maint_high']", "(~x['safety_high']&x['lug_boot_small'])", "~x['maint_high']", "~x['maint_low']", "~x['buying_vhigh']", "x['lug_boot_small']", "(x['maint_vhigh']&~x['maint_high'])", "x['safety_med']", "~x['persons_more']&(x['persons_4']&x['doors_3'])"]
```

From here, one can extract all kinds of statistics concerning the phenotype of the best individuals.

## Building a Custom Operator

Here we will build and use the "MyOpt" operator, showcased in the [Representation and Operators](walkthrough.md#representation-and-operators) section, and the concatLen operator to concat 2 strings and return the final length.

Simply create the MyOpt and concatLen functions and wrap them with the Infix class:

```python
from fedora.sge.utilities.protected_math import Infix

def MyOpt(series: pd.Series):
    return series.apply(lambda x: x.count("i"))

def concatLen(s1: pd.Series, s2: pd.Series):
    return pd.Series(s1 + s2).apply(lambda x: x[::-1])

_MyOpt_ = Infix(MyOpt)
_cclen_ = Infix(concatLen)
```

One can invoke the functions in the following ways:

```python
# Arity 1:
print(_MyOpt_(...))            # Simple Call
print(_MyOpt_ | ...)           # with __or__
print(_MyOpt_ >> ...)          # with __rshift__

# Arity 2:
print(_cclen_(..., ...))       # Simple Call
print(... | _cclen_ | ...)     # with __or__ and __ror__
print(... << _cclen_ >> ...)   # with __rshift__ and __rlshift__
```

This is very useful since it allows for function calls without needing to enumerate the arguments with commas. When evaluating the operators using actual features, we obtain:

```python
features = pd.DataFrame({
    "feature0": pd.Series(["Test i", "Test ii", "Test iii"]),
    "feature1": pd.Series(["Hello", "World", "!"])
})
labels = [0, 0, 1]
```

```python
print((_MyOpt_(df["feature0"])).values)
print((_MyOpt_ | df["feature0"]).values)
print((_MyOpt_ >> df["feature0"]).values)

print(_cclen_(df["feature0"], df["feature1"]).values)
print((df["feature0"] | _cclen_ | df["feature1"]).values)
print((df["feature0"] << _cclen_ >> df["feature1"]).values)
```

```
[1 2 3]
[1 2 3]
[1 2 3]
[11 12  9]
[11 12  9]
[11 12  9]
```

Therefore, the operators function as intended.

The next step is to incorporate these operators into the grammar, which might appear as follows:

```bnf
# strings.pybnf
<start> ::= <feature>,|<feature>,<feature>|<feature>,<feature>,<feature>
<feature> ::=  _MyOpt_ \eb x[<var>] | x[<var>] \l\l _cclen_ \g\g x[<var>] 
<var> ::= 'feature0' | 'feature1'
```

The final step is to incorporate these operators into the framework.

```python
fedora = Fedora(
    seed = 105,
    model = DecisionTreeClassifier(),
    error_metric = errorBAcc,
    sge_parameters_path = "strings.yml",
    grammar_path = "strings.pybnf",
    logging_dir = "./",
    operators = {"_MyOpt_": _MyOpt_, "_cclen_": _cclen_}
)
```

```python
new_features = fedora.fit_transform(features, labels)
print(new_features, end="\n\n")

best = Fedora.get_best("strings-results/", SEED)
features = get_features(best["phenotype"])

for i, feature in enumerate(features):
    print(f"feature_{i}: {feature}")
```

```
   feature_0  feature_1  feature_2
0       12.0       10.0        0.0
1       14.0       10.0        0.0
2       16.0        2.0        0.0

feature_0: x['feature0'] << _cclen_ >> x['feature0']
feature_1: x['feature1'] << _cclen_ >> x['feature1']
feature_2: _MyOpt_ | x['feature1']
```

## Ephemeral Constants

We also provide support for [ephemeral constants](https://deap.readthedocs.io/en/master/api/gp.html) via the 'RANDFLOAT' and 'RANDINT' built-in functions. E.g. Supposing there are two types of numerical data — float and int — random noise can be incorporated in the following manner:

```bnf
...
<flt> ::= <flt><op><flt> | x[<fvar>] | x[<fvar>] + <fnoise>
<int> ::= <int><op><int> | x[<ivar>] | x[<ivar>] + <inoise>
<op> ::= +|-
<fnoise> ::= RANDFLOAT(0, 10) 
<inoise> ::= RANDINT(0, 10) 
...
```

Where the values of **fnoise** and **inoise** are ephemeral constants.

## Tailored Setups

If additional customization of the framework is required, such as incorporating a penalty into the fitness function, implementing early stopping for the evolution process, or logging more generational information, users can create their own class by inheriting from the Fedora class:

```python
import warnings

from fedora.core.engine import Fedora
from fedora.core.utilities.lib import get_features

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

    # Adding bests number of features to progress_report.csv
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
```

This example is available in [here](https://github.com/miguelrabuge/fedora/tree/main/examples/custom-mnist/main.py).

## Structured Grammatical Evolution

Structured Grammatical Evolution is also available through this package. Here is a symbolic regression example:

```PYTHON
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
```
