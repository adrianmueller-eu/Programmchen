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
    a[np.isnan(a)] = 0
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def matexp(A0, eps=1e-5):
    from math import factorial

    # there is a faster method for symmetric matrices
    if is_symmetric(A0):
        eigval, eigvec = np.linalg.eig(A0)
        return eigvec @ np.diag(np.exp(eigval)) @ eigvec.T

    Asum = np.zeros(A0.shape, dtype=float)
    Asum_prev = None

    for i in range(0,10000):
        if i == 0:
            A = np.eye(A0.shape[0]) # A^0 = I
        else:
            A = np.array(factorial(i-1) * A @ A0 / factorial(i), dtype=float)
        Asum_prev = Asum.copy()
        Asum += A
        if np.sum(np.abs(Asum - Asum_prev)) < eps:
            break # return when converged

    return Asum
