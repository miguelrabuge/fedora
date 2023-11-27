import numpy as np
import pandas as pd

def filler(func):
    def wrap(*args, **kwargs):
        return_type = type(args[0])
        series_args = [pd.Series(arg) for arg in args]
        result: pd.Series = func(*series_args, **kwargs)
        return return_type(result.astype(np.float64).replace([np.nan, np.inf, -np.inf], [1, np.finfo(np.float32).max, np.finfo(np.float32).min]))
    return wrap

@filler
def _log_(x):
    if (x <= 0).all(): 
        return pd.Series([0] * len(x))
    return np.log(x)

@filler
def _sig_(x):
    return 1.0 / (1.0 + _exp_(-x))

@filler
def protdiv(x, y):
    if (y == 0).all():
        return pd.Series([1] * len(x))
    return np.divide(x, y)

@filler
def _exp_(x):
    try:
        return np.exp(x)
    except ValueError:
        return pd.Series([1] * len(x))

@filler
def _inv_(x):
    if (x == 0).all(): 
        return pd.Series([1] * len(x))
    return 1.0 / x

@filler
def _sqrt_(x):
    return np.sqrt(np.abs(x))


class Infix:
    def __init__(self, function):
        self.function = function
        self.__pandas_priority__ = 5000 # To overwrite the pandas operator overloading

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


_div_ = Infix(protdiv)

if __name__ == '__main__':
    print(8 | _div_ | 2)
    print(9.0 | _div_ | 2)
    print(8 | _div_ | 0)
    print(8 / 0)

