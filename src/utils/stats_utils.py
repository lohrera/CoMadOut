import numpy as np

def kurtosis(x, axis=0):
    eps=1e-6
    mu = np.mean(x, axis=axis)
    var = np.var(x, axis=axis)
    var_squared = np.power(var, 2)        
    sc = np.mean(np.power((x - mu), 4), axis=axis) / np.power(np.mean(np.power((x - mu), 2), axis=axis), 2)
    return sc

def _first(arr, axis):
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

def zscore(a, axis=0, ddof=0):
    scores = a.copy()
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=ddof, keepdims=True)
    if axis is None:
        isconst = (a.item(0) == a).all()
    else:
        isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)

    std[isconst] = 1.0
    z = (scores - mn) / std
    return z
