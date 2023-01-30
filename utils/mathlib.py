import sys
import numpy as np
from math import factorial, gcd, log, sqrt
from numpy.random import randint
from itertools import combinations

Phi = (1 + np.sqrt(5))/2

def Fibonacci(n):
    Psi = 1 - Phi
    return (Phi**n - Psi**n)/(Phi - Psi) # /np.sqrt(5)

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

def is_orthogonal(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.T, np.eye(a.shape[0])
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
def series(f, verbose=False, max_iter=100000):
    res = 0.0

    for i in range(max_iter):
        res_i = float(f(i))
        res += res_i
        if verbose:
            print(i, res, res_i)
        if res_i < sys.float_info.epsilon:
            return res # return when converged
        if res == np.inf:
            break

    raise ValueError("Series didn't converge!")

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

def binFrac_i(j, i):
    return int(j % (1/2**(i-1)) != j % (1/2**i))
    #return int(np.ceil((j % (1/2**(i-1))) - (j % (1/2**i))))

def binFrac(j, prec=20):
    return "." + "".join([str(binFrac_i(j,i)) for i in range(1,prec+1)])

def binstr_from_float(f, r=None):
    """
    Convert a float `f` to a binary string with `r` bits after the comma.
    If `r` is None, the number of bits is chosen such that the float is
    represented exactly.
    If `f` is negative, the positive number modulus `2**k` is returned,
    where `k` is the smallest integer such that `2**k > -f`.

    Parameters
    ----------
    f : float
        The float to convert.
    r : int, optional
        The number of bits after the comma. The default is None.

    Returns
    -------
    str
        The binary string.
    """
    # special treatment for negative numbers > -1
    if f < 0 and f > -1:
        if r is not None and f > -1/2**(r+1):
            return '.' + '0'*r
        f = 1+f

    i = 0 # number of bits after the comma
    while int(f) != f:
        if i == r:
            f = int(np.round(f))
            break
        f *= 2
        i += 1
    # if f is negative, find the positive number modulus 2**k,
    # where k is the smallest integer such that 2**k > -f
    if f < 0:
        k = 0
        while -f > 2**(k-1):
            k += 1
        f = 2**k + f
    as_str = str(bin(int(f))).replace('b', '0')
    if i == 0:
        if r is None or r == 0:
           return as_str[2:]
        if as_str[2:] == '0':
            return '.' + '0'*r
        return as_str[2:] + '.' + '0'*r

    before_comma = as_str[2:-i]
    after_comma = '0'*(i-len(as_str[-i:])) + as_str[-i:]
    if r is None:
       return as_str[2:-i] + '.' + after_comma
    return as_str[2:-i] + '.' + after_comma[:r] + '0'*(r-len(after_comma[:r]))

def float_from_binstr(s):
    s = s.split('.')

    pre = 0
    frac = 0
    if len(s[0]) > 0:
        pre = int(s[0], 2)
    if len(s) > 1 and len(s[1]) > 0:
        frac = int(s[1], 2) / 2.**len(s[1])
    return float(pre + frac)

def binstr_from_int(n, places=0):
    return ("{0:0" + str(places) + "b}").format(n)

def int_from_binstr(s):
    return int(float_from_binstr(s))

def int_from_bincoll(l):
    #return sum([2**i*v_i for i,v_i in enumerate(reversed(l))])
    return int_from_binstr(binstr_from_bincoll(l))

def bincoll_from_binstr(s):
    return [int(x) for x in s]

def binstr_from_bincoll(l):
    return "".join([str(x) for x in l])

def prime_factors(n):
    """Simple brute-force algorithm to find prime factors"""
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def is_prime(n, alpha=1e-20): # only up to 2^54 -> alpha < 1e-16.26 (-> 55 iterations; < 1e-20 is 67 iterations)
    """Miller-Rabin test for primality."""
    if n == 1 or n == 4:
        return False
    if n == 2 or n == 3:
        return True

    def getKM(n):
        k = 0
        while n % 2 == 0:
            k += 1
            n /= 2
        return k,int(n)

    p = 1
    while p > alpha:
        a = randint(2,n-2)
        if gcd(a,n) != 1:
            #print(n,"is not prime (1)")
            return False
        k,m = getKM(n-1)
        b = pow(a, m, n)
        if b == 1:
            p *= 1/2
            continue
        for i in range(1,k+1):
            b_new = pow(b,2,n)
            # first appearance of b == 1 is enough
            if b_new == 1:
                break
            b = b_new
            if i == k:
                #print(n,"is not prime (2)")
                return False
        if gcd(b+1,n) == 1 or gcd(b+1,n) == n:
            p *= 1/2
        else:
            #print(n,"is not prime (3)")
            return False

    # print("%d is prime with alpha=%E (if Carmichael number: alpha=%f)" % (n, p, (3/4)**log(p,1/2)))
    return True

def closest_prime_factors_to(n, m):
    """Find the set of k prime factors of n with product closest to m."""
    pf = prime_factors(n)

    min_diff = float("inf")
    min_combo = None
    for k in range(len(pf)):
        for c in combinations(pf, k):
            diff = abs(m - np.prod(c))
            if diff < min_diff:
                min_diff = diff
                min_combo = c
    return min_combo

def closest_prime_factors_to_sqrt(n):
    """Find the set of k prime factors of n with product closest to sqrt(n)."""
    return closest_prime_factors_to(n, sqrt(n))
