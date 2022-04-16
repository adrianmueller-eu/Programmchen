import sys
import numpy as np

Phi = (1 + np.sqrt(5))/2

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    a = np.array(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False
    a[np.isnan(a)] = 0
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_hermitian(a, rtol=1e-05, atol=1e-08):
    a = np.array(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False
    a[np.isnan(a)] = 0
    return np.allclose(a, a.conj().T, rtol=rtol, atol=atol)

def is_complex(a):
    return np.iscomplex(a).any()
#    return a.dtype == "complex128"

def deg(x):
    return x/np.pi*180

def rad(x):
    return x/180*np.pi

def matexp(A0):
    from math import factorial

    # there is a faster method for hermitian matrices
    if is_hermitian(A0):
        eigval, eigvec = np.linalg.eig(A0)
        return eigvec @ np.diag(np.exp(eigval)) @ eigvec.conj().T

    Asum = np.zeros(A0.shape, dtype=A0.dtype)
    Asum_prev = None

    for i in range(0,10000):
        if i == 0:
            A = np.eye(A0.shape[0]) # A^0 = I
        else:
            A = np.array(factorial(i-1) * A @ A0 / factorial(i), dtype=A0.dtype)
        Asum_prev = Asum.copy()
        Asum += A
        if np.sum(np.abs(Asum - Asum_prev)) < sys.float_info.epsilon:
            return Asum # return when converged

    raise ValueError("Convergence failed! (try scipy.linalg.expm instead)")

# e.g. series(lambda n: 1/factorial(2*n)) + series(lambda n: 1/factorial(2*n + 1))
def series(f, pr=False, max_iter=100000):
    res = 0.0

    for i in range(max_iter):
        res_i = float(f(i))
        res += res_i
        if pr:
            print(i, res, res_i)
        if res_i < sys.float_info.epsilon:
            return res # return when converged
        if res == np.inf:
            break

    raise ValueError("Series doesn't converge!")
