import sys
import numpy as np
from math import factorial

Phi = (1 + np.sqrt(5))/2

def _sq_matrix_allclose(a, f, rtol=1e-05, atol=1e-08):
    a = np.array(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False
    a[np.isnan(a)] = 0
    a, b = f(a)
    return np.allclose(a, b, rtol=rtol, atol=atol)

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
    	a, a.T
    ), rtol=rtol, atol=atol)

def is_hermitian(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
    	a, a.conj().T
    ), rtol=rtol, atol=atol)

def is_unitary(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
    	a @ a.conj().T, np.eye(a.shape[0])
    ), rtol=rtol, atol=atol)

def is_involutory(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
    	a @ a, np.eye(a.shape[0])
    ), rtol=rtol, atol=atol)

def is_complex(a):
    if hasattr(a, 'dtype'):
        return a.dtype == complex
    return np.iscomplex(a).any()
#    return a.dtype == "complex128"

def deg(x):
    return x/np.pi*180

def rad(x):
    return x/180*np.pi

try:
    from scipy.linalg import expm as matexp
except:
    def matexp(A0):
        # there is a faster method for hermitian matrices
        if is_hermitian(A0):
            eigval, eigvec = np.linalg.eig(A0)
            return eigvec @ np.diag(np.exp(eigval)) @ eigvec.conj().T

        Asum = np.zeros(A0.shape, dtype=A0.dtype)

        for i in range(0,10000):
            if i == 0:
                A = np.eye(A0.shape[0]) # A^0 = I
            else:
                A = np.array(factorial(i-1) * A @ A0 / factorial(i), dtype=A0.dtype)
            Asum += A
            if np.sum(np.abs(A)) < sys.float_info.epsilon:
                return Asum # return when converged

        raise ValueError("Convergence failed! (try scipy.linalg.expm instead)")

    def matlog(A):
        evals, evecs = np.linalg.eig(A)
        return evecs @ np.diag(np.log(evals.astype(complex))) @ evecs.conj().T

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

def normalize(a, p=2, remove_global_phase=True):
     if is_complex(a):
         a = np.array(a, dtype=complex)
     else:
         a = np.array(a, dtype=float)
     a /= np.linalg.norm(a, ord=p)
     if remove_global_phase and is_complex(a):
         a *= np.exp(-1j*np.angle(a[0]))
     return a

def choice(a, size=None, replace=True, p=None):
    if p is not None:
        if np.abs(np.sum(p) - 1) > sys.float_info.epsilon:
            p = normalize(p, p=1)

    if hasattr(a, '__len__'):
        n = len(a)
        idx = np.random.choice(n, size=size, replace=replace, p=p)
        return np.array(a)[idx]
    else:
        return np.random.choice(a, size=size, replace=replace, p=p)
