import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA = "X"
LABELS = "y"

TRAIN = "train"
TEST = "test"

FITNESS = "fitness"
PHENOTYPE = "phenotype"
FEATURES = "features"


def format_data(X, y):
    return {DATA: X, LABELS: y}

def bound_dataset(df: pd.DataFrame, precision=np.float32):
    return df.where(df >= np.finfo(precision).min, np.finfo(precision).min)   \
             .where(df <= np.finfo(precision).max, np.finfo(precision).max)   \
             .replace([np.nan], 1)                                            \
             .astype(precision)

def scale_data(train, test):
    sc = StandardScaler()
    
    ttrain = sc.fit_transform(train.astype(np.float64))
    ttest = sc.transform(test.astype(np.float64))
    
    df_train = pd.DataFrame(ttrain, columns=train.columns)
    df_test = pd.DataFrame(ttest, columns=test.columns)

    return bound_dataset(df_train), bound_dataset(df_test)

def split_data(x, y, seed, test_size=0.5):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    return  format_data(pd.DataFrame(X_train.values, columns=X_train.columns), y_train),                 \
            format_data(pd.DataFrame(X_test.values, columns=X_test.columns), y_test),                    \

def engineer_dataset(phenotype, data, globals=globals()):
    """ Phenotype = Tuple: (Func1, Func2, ... , FuncN)"""
    # Dataset = Tuple: (pd.Series(NewFeature1), ..., pd.Series(NewFeatureN))
    new_dataset: tuple = eval(phenotype, globals, {"x": data[DATA]})
    
    # Dataset = pd.DataFrame: [feature_1: pd.Series(NewFeature1), ... , feature_n: pd.Series(NewFeatureN)] 
    new_dataset: pd.DataFrame = pd.DataFrame({f"feature_{i}": series.to_list() for i, series in enumerate(new_dataset)})
    
    # Set min and max bounds and format data
    return format_data(bound_dataset(new_dataset), data[LABELS])

def score(model, train, test, metric):
    # Scale Data
    X_train, X_test = scale_data(train[DATA], test[DATA])

    # Fit Model
    predictions = model.fit(X_train, train[LABELS]).predict(X_test)
    score = metric(test[LABELS], predictions)
    return score

def get_features(phenotype: str):
    features = list(filter(lambda x: x != "", phenotype.split(",")))
    return features

def generate_grammar(max_features=10, operators1=None, operators2=["+", "-", "*", "/"], columns=["c0", "c1", "c2"]):
    start = "<start> ::= <feature>," + "".join(["|<feature>" + ",<feature>"*i for i in range(1, max_features)])
    feature = "<feature> ::= <feature><op2><feature> | (<feature><op2><feature>) | x[<var>]"
    if operators1:
        feature += " | <op1>x[<var>]"
        op1 = "<op1> ::= "+ "|".join(operators1)
    op2 = "<op2> ::= " + "|".join(operators2)
    var = "<var> ::= " + "|".join([f"'{col}'" if type(col) == str else str(col) for col in columns])
    return "\n".join([start, feature, op1, op2, var])
