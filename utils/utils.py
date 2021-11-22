import numpy as np

def moving_avg(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def r(x, precision=5):
    return round(x, precision)

def bins_sqrt(data):
    return int(np.ceil(np.sqrt(len(data))))

def logbins(data, start=None, stop=None, num=None, scale=2):
    if start is None:
        start = min(data)/scale
    if stop is None:
        stop = max(data)*scale
    if num is None:
        num = bins_sqrt(data)
    return 10**(np.linspace(np.log10(start),np.log10(stop),num))

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    a = np.array(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
