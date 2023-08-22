import sys
import numpy as np
from math import factorial, sqrt
from itertools import combinations, chain
import scipy.sparse as sp

### Tests

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

def is_psd(a, rtol=1e-05, atol=1e-08):
    if not is_hermitian(a, rtol=rtol, atol=atol):
        return False
    eigv = np.linalg.eigvalsh(a)
    return np.all(np.abs(eigv.imag) < atol) and np.all(eigv.real >= -atol)

### Conversion

def deg(rad):
    return rad/np.pi*180

def rad(deg):
    return deg/180*np.pi


### Functions

# e.g. series(lambda n, _: 1/factorial(2*n)) + series(lambda n, _: 1/factorial(2*n + 1))
def series(f, start_value=0, start_index=0, eps=sys.float_info.epsilon, max_iter=100000, verbose=False):
    if not np.isscalar(start_value):
        start_value = np.array(start_value)
    res = start_value
    res_i = res
    for n in range(start_index, max_iter):
        res_i = f(n, res_i)
        res += res_i
        if verbose:
            print(f"Iteration {n}:", res, res_i)
        if np.sum(np.abs(res_i)) < eps:
            return res # return when converged
        if np.max(res) == np.inf:
            break

    raise ValueError("Series didn't converge!")

try:
    from scipy.linalg import expm as matexp
    from scipy.linalg import logm as _matlog
    from scipy.linalg import sqrtm as matsqrt
    from scipy.linalg import fractional_matrix_power as matpow

    def matlog(A, base=np.e):
        return _matlog(A) / np.log(base)
except:
    def matexp(A0, base=np.e):
        # there is a faster method for hermitian matrices
        if is_hermitian(A0):
            eigval, eigvec = np.linalg.eig(A0)
            return eigvec @ np.diag(np.power(base, eigval)) @ eigvec.conj().T
        # use series expansion
        return np.eye(A0.shape[0]) + series(lambda n, A: A @ A0 / n, start_value=A0, start_index=2)

    def matlog(A, base=np.e):
        evals, evecs = np.linalg.eig(A)
        return evecs @ np.diag(np.log(evals.astype(complex)) / np.log(base)) @ evecs.conj().T

    def matpow(A, n):
        evals, evecs = np.linalg.eig(A)
        return evecs @ np.diag(evals.astype(complex)**n) @ evecs.conj().T

    def matsqrt(A, n=2):
        return matpow(A, 1/n)

def normalize(a, p=2, remove_global_phase=True):
     if is_complex(a):
         a = np.array(a, dtype=complex)
     else:
         a = np.array(a, dtype=float)
     a /= np.linalg.norm(a, ord=p)
     if remove_global_phase and is_complex(a):
         a *= np.exp(-1j*np.angle(a[0]))
     return a

def softmax(a, beta=1):
     a = np.exp(beta*a)
     Z = np.sum(a)
     return a / Z


### Sets

# https://docs.python.org/3/library/itertools.html
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

### Groups

def SO(n):
    """ Special orthogonal group. Returns n(n-1)/2 functions that take an angle and return the corresponding real rotation matrix """
    def rotmat(i, j, phi):
        a = np.eye(n)
        a[i,i] = np.cos(phi)
        a[j,j] = np.cos(phi)
        a[i,j] = -np.sin(phi)
        a[j,i] = np.sin(phi)
        return a
    return [lambda phi: rotmat(i, j, phi) for i,j in combinations(range(n), 2)]

def su(n, include_identity=False, sparse=False):
    """ The Lie algebra associated with the Lie group SU(n). Returns the n^2-1 generators (traceless Hermitian matrices) of the group. Use `include_identity = True` to return a complete orthogonal basis of hermitian `n x n` matrices.

    Parameters
        n (int): The dimension of the matrices.
        include_identity (bool, optional): If True, include the identity matrix in the basis (default: False).
        sparse (bool, optional): If True, return a sparse representation of the matrices (default: False).

    Returns
        list[ np.ndarray | scipy.sparse.csr_array ]: A list of `n^2-1` matrices that form a basis of the Lie algebra.
    """
    if sparse:
        base = sp.lil_array((n,n), dtype=complex)
    else:
        if n > 100:
            print(f"Warning: For `n = {n} > 100`, it is recommended to use `sparse=True` to save memory.")
        base = np.zeros((n,n), dtype=complex)

    basis = []
    # Generate the off-diagonal matrices
    for i in range(n):
        for j in range(i+1, n):
            m = base.copy()
            m[i,j] = 1
            m[j,i] = 1
            basis.append(m)

            m = base.copy()
            m[i, j] = -1j
            m[j, i] = 1j
            basis.append(m)

    # Generate the diagonal matrices
    for i in range(1,n):
        m = base.copy()
        for j in range(i):
            m[j,j] = 1
        m[i,i] = -i
        if i > 1:
            m = np.sqrt(2/(i*(i+1))) * m
        basis.append(m)

    if include_identity:
        identity = base.copy()
        for i in range(n):
            identity[i,i] = 1
        basis.append(identity)

    if sparse:
        # convert to csr format for faster arithmetic operations
        return [sp.csr_matrix(m) for m in basis]
    return basis

def SU(n):
    """ Special unitary group. Returns n^2-1 functions that take an angle and return the corresponding complex rotation matrix """
    generators = su(n)
    def rotmat(G):
        return lambda phi: matexp(-1j*phi/2*G)
    return [rotmat(G) for G in generators]

### Random

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

def random_vec(size, limits=(0,1), complex=False):
    if complex:
        return random_vec(size, limits=limits) + 1j*random_vec(size, limits=limits)
    return np.random.uniform(limits[0], limits[1], size=size)

def random_symmetric(size):
    if not hasattr(size, '__len__'):
        size = (size, size)
    a = random_vec(size)
    return (a + a.T)/2

def random_orthogonal(size):
    if not hasattr(size, '__len__'):
        size = (size, size)
    a = random_vec(size)
    q, r = np.linalg.qr(a)
    return q

def random_hermitian(size):
    if not hasattr(size, '__len__'):
        size = (size, size)
    a = random_vec(size, complex=True)
    return (a + a.conj().T)/2

def random_unitary(size):
    if not hasattr(size, '__len__'):
        size = (size, size)
    a = random_hermitian(size)
    return matexp(1j*a)

def random_psd(size):
    if not hasattr(size, '__len__'):
        size = (size, size)
    a = random_vec(size, complex=True)
    return a @ a.conj().T

### Integers & Primes

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

def closest_prime_factors_to(n, m):
    """Find the set of prime factors of n with product closest to m."""
    pf = prime_factors(n)

    min_diff = float("inf")
    min_combo = None
    for c in powerset(pf):
        diff = abs(m - np.prod(c))
        if diff < min_diff:
            min_diff = diff
            min_combo = c
    return min_combo

def int_sqrt(n):
    """For integer $n$, find the integer $a$ closest to $\sqrt{n}$, such that $n/a$ is also an integer."""
    if n == 1 or n == 0:
        return n
    return int(np.prod(closest_prime_factors_to(n, sqrt(n))))


### Binary strings

def binFrac_i(j, i):
    return int(j % (1/2**(i-1)) != j % (1/2**i))
    #return int(np.ceil((j % (1/2**(i-1))) - (j % (1/2**i))))

def binFrac(j, prec=20):
    return "." + "".join([str(binFrac_i(j,i)) for i in range(1,prec+1)])

def binstr_from_float(f, r=None, complement=False):
    """
    Convert a float `f` to a binary string with `r` bits after the comma.
    If `r` is None, the number of bits is chosen such that the float is
    represented exactly.

    Parameters
        f (float): The float to convert.
        r (int), optional: The number of bits after the comma. The default is None.
        complement (bool), optional: If True and `f < 0`, count the fraction "backwards" (e.g. -0.125 == '-.111').

    Returns
        str: The binary string representing `f`.
    """
    negative = f < 0
    if negative:
        f = -f # make it easier to handle the minus sign in the end
        if r is not None and r > 0 and abs(f) < 1/2**(r+1):
            return '.' + '0'*r
        if complement:
            # Translate the fraction to the corresponding complement, e.g. -0.125 => -0.875
            # alternatively, we could also flip all bits in `frac_part` below and add 1
            frac = f - int(f)
            if frac > 0:
                f = int(f) - frac + 1  # -1 if f was negative

    i = 0 # number of bits in the fraction part
    while int(f) != f:
        if r is not None and i >= r:
            f = int(np.round(f))
            break
        f *= 2
        i += 1
    f = int(f) # there should be no fractional part left

    # We use complement only for the fraction, not for the integer part
    # # If `f` is negative, the positive number modulus `2**k` is returned,
    # # where `k` is the smallest integer such that `2**k > -f`.
    # if f < 0:
    #     k = 0
    #     while -f > 2**(k-1):
    #         k += 1
    #     f = 2**k + f

    # integer part
    as_str = str(bin(f))[2:] # this adds a leading '-' sign for negative numbers
    sign = '-' if negative else ''
    # print(f, i, sign, as_str)
    if i == 0: # no fraction part
        if r is None or r <= 0: # ==> i == 0
            return sign + as_str
        if as_str == '0':
            return sign + '.' + '0'*r
        return sign + as_str + '.' + '0'*r
    int_part = sign + as_str[:-i]

    # fraction part
    frac_part = '0'*(i-len(as_str)) + as_str[-i:]
    # print(int_part, frac_part)
    if r is None:
       return int_part + '.' + frac_part
    return int_part + '.' + frac_part[:r] + '0'*(r-len(frac_part[:r]))

def float_from_binstr(s, complement=False):
    """ Convert a binary string to a float.

    Parameters
        s (str): The binary string.
        complement (bool, optional): If True, interpret the fraction part as the complement of the binary representation. Defaults to False.

    Returns
        float: The float represented by the binary string.
    """

    negative = s[0] == '-'
    if negative:
        s = s[1:]
    s = s.split('.')

    pre = 0
    frac = 0
    if len(s[0]) > 0:
        pre = int(s[0], 2)
    if len(s) > 1 and len(s[1]) > 0:
        if negative and complement:
            # flip all bits and add 1
            s[1] = ''.join(['1' if x == '0' else '0' for x in s[1]])
            frac = int(s[1], 2) + 1
        else:
            frac = int(s[1], 2)
        frac /= 2.**len(s[1])
    return float(pre + frac) * (-1 if negative else 1)

def binstr_from_int(n, places=0):
    return ("{0:0" + str(places) + "b}").format(n)

def int_from_binstr(s):
    return int(float_from_binstr(s))

def bincoll_from_binstr(s):
    return [int(x) for x in s]

def binstr_from_bincoll(l):
    return "".join([str(x) for x in l])

def int_from_bincoll(l):
    #return sum([2**i*v_i for i,v_i in enumerate(reversed(l))])
    return int_from_binstr(binstr_from_bincoll(l))

def bincoll_from_int(n, places=0):
    return bincoll_from_binstr(binstr_from_int(n, places))


### Misc

Phi = (1 + np.sqrt(5))/2
def Fibonacci(n):
    Psi = 1 - Phi
    return int(np.round((Phi**n - Psi**n)/(Phi - Psi))) # /np.sqrt(5)

def calc_pi(N=3):
    from decimal import Decimal, getcontext
    getcontext().prec = 14*N
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(factorial(6*n)*(13591409+545140134*n))/Decimal(factorial(3*n)*(factorial(n)*(-640320)**n)**3)
        r += r_i
    return Decimal(4270934400)/(Decimal(10005).sqrt()*r)

def calc_pi2(N=5):
    from decimal import Decimal, getcontext
    getcontext().prec = 8*N
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(factorial(4*n)*(1103+26390*n))/Decimal((factorial(n)*396**n)**4)
        r += r_i
    return Decimal(9801)/(Decimal(8).sqrt()*r)

def sqrt_brain_compatible(x, correction_term=False, n_max = 20):
    """ Nice way to calculate approximate square roots in the head:
        1. Find the largest integer n, such that n² < x
        2. sqrt(x) ≈ n + (x-n²)/(2n+1) + 1/(6(2n+1))
    The last term is more negligible the larger x.
        """
    n_sq_table = [n**2 for n in range(1,n_max)]
    if x <= 1:
       raise ValueError("Are you kidding me")
    if x > n_sq_table[-1]:
       raise ValueError("Use a proper function for this, please.")
    for n,n2 in enumerate(n_sq_table):
       if n2 > x:
           if correction_term:
               return n + (x - n_sq_table[n-1])/(2*n + 1) + 1/(6*(2*n+1))
           else:
               return n + (x - n_sq_table[n-1])/(2*n + 1)

# Reduction of the halting problem to the equivalence problem, i.e. show that the latter is at least as hard as the former. Give an algorithm `H` and input `x` to decide whether `H(x)` halts. This function returns two functions `f1, f2`. Use your implementation `equiv` solving the equivalence problem to solve the halting problem, e.g. `equiv(*equiv_from_halt(H, x))`.
def equiv_from_halt(H, x):
    def f1(y):
        if y == x:
            return H(x)
        return True

    def f2(y):
        return True

    return f1, f2

def Hutchinson_trace(A, n=1000):
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    zs = np.random.normal(0, 1, size=(A.shape[0], n))
    return np.mean([z.T @ A @ z for z in zs.T])


### Tests

def test_mathlib_all():
    tests = [
        _test_is_symmetric,
        _test_random_symmetric,
        _test_is_orthogonal,
        _test_random_orthogonal,
        _test_is_hermitian,
        _test_random_hermitian,
        _test_is_unitary,
        _test_random_unitary,
        _test_is_psd,
        _test_random_psd,
        _test_rad,
        _test_deg,
        _test_matexp,
        _test_matlog,
        _test_series,
        _test_series2,
        _test_normalize,
        _test_softmax,
        _test_su,
        _test_SU,
        _test_prime_factors,
        _test_closest_prime_factors_to,
        _test_int_sqrt,
        _test_binFrac,
        _test_binstr_from_float,
        _test_float_from_binstr,
        _test_binstr_from_int,
        _test_int_from_binstr,
        _test_bincoll_from_binstr,
        _test_binstr_from_bincoll,
        _test_int_from_bincoll,
        _test_bincoll_from_int,
        _test_Fibonacci,
        _test_calc_pi
    ]

    for test in tests:
        print("Running", test.__name__, "... ", end="")
        if test():
            print("Test succeed!")
        else:
            print("ERROR!")
            break

def _test_is_symmetric():
    a = np.random.rand(5,5)
    b = a + a.T
    assert is_symmetric(b)

    c = a + 1
    assert not is_symmetric(c)
    return True

def _test_random_symmetric():
    a = random_symmetric(5)
    assert is_symmetric(a)
    return True

def _test_is_orthogonal():
    a, b = np.random.rand(2)
    a, b = normalize([a, b])
    a = np.array([
        [a, b],
        [-b, a]
    ])
    assert is_orthogonal(a)

    c = a + 1
    assert not is_orthogonal(c)
    return True

def _test_random_orthogonal():
    a = random_orthogonal(5)
    assert is_orthogonal(a)
    return True

def _test_is_hermitian():
    a = np.random.rand(5,5) + 1j*np.random.rand(5,5)
    b = a + a.conj().T
    assert is_hermitian(b)
    c = a + 1
    assert not is_hermitian(c)
    return True

def _test_random_hermitian():
    a = random_hermitian(5)
    assert is_hermitian(a)
    return True

def _test_is_unitary():
    a, b = np.random.rand(2) + 1j*np.random.rand(2)
    a, b = normalize([a, b], remove_global_phase=False)
    phi = np.random.rand()*2*np.pi
    a = np.array([
        [a, b],
        [-np.exp(1j*phi)*b.conjugate(), np.exp(1j*phi)*a.conjugate()]
    ])
    assert is_unitary(a)

    c = a + 1
    assert not is_unitary(c)
    return True

def _test_random_unitary():
    a = random_unitary(5)
    assert is_unitary(a)
    return True

def _test_is_psd():
    # A @ A^\dagger => PSD
    a = random_vec((5,5), complex=True)
    a = a @ a.conj().T
    assert is_psd(a)

    # unitarily diagonalizable (= normal) + positive eigenvalues <=> PSD
    U = random_unitary(5)
    p = np.random.rand(5)
    a = U @ np.diag(p) @ U.conj().T
    assert is_psd(a)

    # sum(p) can't be larger than 5 here, so make the trace negative to guarantee negative eigenvalues
    b = a - 5
    assert not is_psd(b)
    return True

def _test_random_psd():
    a = random_psd(5)
    assert is_psd(a)
    return True

def _test_rad():
    assert rad(180) == np.pi
    assert rad(0) == 0
    return True

def _test_deg():
    assert deg(np.pi) == 180
    assert deg(0) == 0
    return True

def _test_matexp():
    a = np.random.rand(5,5) + 1j*np.random.rand(5,5)
    # check if det(matexp(A)) == exp(trace(A))
    assert np.isclose(np.linalg.det(matexp(a)), np.exp(np.trace(a)))
    return True

def _test_matlog():
    alpha = np.random.rand()*2*np.pi - np.pi
    A = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
    assert np.allclose(matlog(A), alpha*np.array([[0, -1],[1, 0]])), f"Error for alpha = {alpha}! {matlog(A)} != {alpha*np.array([[0, -1],[1, 0]])}"
    return True

def _test_series():
    a = series(lambda n, _: 1/factorial(2*n)) + series(lambda n, _: 1/factorial(2*n + 1))
    assert np.isclose(a, np.e)
    return True

def _test_series2():
    # pauli X
    A0 = np.array([[0, 1.], [1., 0]])
    a = np.eye(A0.shape[0]) + series(lambda n, A: A @ A0 / n, start_value=A0, start_index=2)
    expected = np.array([[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]])
    assert np.allclose(a, expected)

    # pauli Z
    A0 = np.array([[1., 0], [0, -1.]])
    a = np.eye(A0.shape[0]) + series(lambda n, A: A @ A0 / n, start_value=A0, start_index=2)
    expected = np.array([[np.e, 0], [0, 1/np.e]])
    assert np.allclose(a, expected)
    return True

def _test_normalize():
    a = np.random.rand(5) + 1j*np.random.rand(5)
    b = normalize(a)
    assert np.isclose(np.linalg.norm(b), 1)
    return True

def _test_softmax():
    a = np.random.rand(5)
    b = softmax(a)
    assert np.isclose(np.sum(b), 1)
    return True

def _test_su():
    n = 5
    sun = su(n)

    # check the number of generators
    n_expected = n**2-1
    assert len(sun) == n_expected, f"Number of generators is {len(sun)}, but should be {n_expected}!"

    # check if no two generators are the same
    for i, (A,B) in enumerate(combinations(sun,2)):
        assert not np.allclose(A, B), f"Pair {i} is not different!"

    # check if all generators are traceless
    for i, A in enumerate(sun):
        assert np.isclose(np.trace(A), 0), f"Generator {i} is not traceless!"

    # check if all generators are Hermitian
    for i, A in enumerate(sun):
        assert is_hermitian(A), f"Generator {i} is not Hermitian!"

    # check if all generators are orthogonal
    for i, (A,B) in enumerate(combinations(sun,2)):
        assert np.allclose(np.trace(A.conj().T @ B), 0), f"Pair {i} is not orthogonal!"

    # check if all generators have matrix norm sqrt(2)
    for i, A in enumerate(sun):
        assert np.isclose(np.linalg.norm(A), sqrt(2)), f"Generator {i} does not have norm 2!"

    # check sparse representation
    sun_sp = su(n, sparse=True)

    # check the generators are the same
    for i, (A,B) in enumerate(zip(sun, sun_sp)):
        assert np.allclose(A, B.todense()), f"Pair {i} is not the same!"

    return True

def _test_SU():
    n = 4
    SUn = SU(n)

    # check the number of generators
    n_expected = n**2-1
    assert len(SUn) == n_expected, f"Number of generators is {len(SUn)}, but should be {n_expected}!"

    # check if all generators are unitary
    for i, A in enumerate(SUn):
        random_angle = np.random.randn()
        assert is_unitary(A(random_angle)), f"Generator {i} is not unitary! ({random_angle})"

    # check if no two generators are the same
    for i, (A,B) in enumerate(combinations(SUn,2)):
        random_angle = np.random.randn()
        assert not np.allclose(A(random_angle), B(random_angle)), f"Pair {i} is not different! ({random_angle})"

    return True

def _test_prime_factors():
    assert prime_factors(12) == [2, 2, 3] and prime_factors(1) == []
    return True

def _test_closest_prime_factors_to():
    assert np.array_equal(closest_prime_factors_to(42, 13), [2, 7])
    return True

def _test_int_sqrt():
    assert int_sqrt(42) == 6
    assert int_sqrt(1) == 1
    assert int_sqrt(0) == 0
    return True

def _test_binFrac():
    assert binFrac(0.5, prec=12) == ".100000000000"
    assert binFrac(0.5, prec=0) == "."
    assert binFrac(np.pi-3, prec=12) == ".001001000011"
    return True

def _test_binstr_from_float():
    assert binstr_from_float(0) == "0"
    assert binstr_from_float(10) == "1010"
    assert binstr_from_float(0.5) == ".1"
    assert binstr_from_float(0.5, r=12) == ".100000000000"
    assert binstr_from_float(np.pi, r=20) == "11.00100100001111110111"
    assert binstr_from_float(0.5, r=0) == "0"  # https://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers
    assert binstr_from_float(0.50000001, r=0) == "1"
    assert binstr_from_float(-3) == "-11"
    assert binstr_from_float(-1.5, r=3) == "-1.100"
    assert binstr_from_float(-0.125) == "-.001"
    assert binstr_from_float(-0.125, complement=True) == "-.111"
    assert binstr_from_float(-0.875, complement=False) == "-.111"
    assert binstr_from_float(0, r=3, complement=True) == ".000"
    assert binstr_from_float(-1.0, r=3, complement=True) == "-1.000"
    return True

def _test_float_from_binstr():
    assert np.allclose(float_from_binstr('1010'), 10)
    assert np.allclose(float_from_binstr('0'), 0)
    assert np.allclose(float_from_binstr('.100000000000'), 0.5)
    assert np.allclose(float_from_binstr('11.00100100001111110111'), np.pi)
    assert np.allclose(float_from_binstr('-11'), -3)
    assert np.allclose(float_from_binstr('-1.100'), -1.5)
    assert np.allclose(float_from_binstr('-.001'), -0.125)
    assert np.allclose(float_from_binstr('-.111', complement=True), -0.125)
    assert np.allclose(float_from_binstr('-.111', complement=False), -0.875)

    # check consistency of binstr_from_float and float_from_binstr
    assert np.allclose(float_from_binstr(binstr_from_float(0.5, r=2)), 0.5)
    assert np.allclose(float_from_binstr(binstr_from_float(-np.pi, r=20)), -np.pi, atol=1e-6)
    assert np.allclose(float_from_binstr(binstr_from_float(-0.375, r=3)), -0.375)
    assert np.allclose(float_from_binstr(binstr_from_float(-0.375, r=3, complement=True), complement=True), -0.375)
    return True

def _test_binstr_from_int():
    assert binstr_from_int(42) == "101010"
    assert binstr_from_int(0) == "0"
    assert binstr_from_int(1) == "1"
    return True

def _test_int_from_binstr():
    assert int_from_binstr("101010") == 42
    assert int_from_binstr("0") == 0
    assert int_from_binstr("1") == 1
    return True

def _test_binstr_from_bincoll():
    assert binstr_from_bincoll([1, 0, 1, 0, 1, 0]) == "101010"
    assert binstr_from_bincoll([0]) == "0"
    assert binstr_from_bincoll([1]) == "1"
    return True

def _test_bincoll_from_binstr():
    assert bincoll_from_binstr("101010") == [1, 0, 1, 0, 1, 0]
    assert bincoll_from_binstr("0") == [0]
    assert bincoll_from_binstr("1") == [1]
    return True

def _test_int_from_bincoll():
    assert int_from_bincoll([1, 0, 1, 0, 1, 0]) == 42
    assert int_from_bincoll([0]) == 0
    assert int_from_bincoll([1]) == 1
    return True

def _test_bincoll_from_int():
    assert bincoll_from_int(42) == [1, 0, 1, 0, 1, 0]
    assert bincoll_from_int(0) == [0]
    assert bincoll_from_int(1) == [1]
    return True

def _test_Fibonacci():
    assert Fibonacci(10) == 55
    assert Fibonacci(0) == 0
    assert Fibonacci(1) == 1
    return True

def _test_calc_pi():
    pi_str = "3.1415926535897932384626433832795028841971"
    assert str(calc_pi(3))[:42] == pi_str
    assert str(calc_pi2(6))[:42] == pi_str
    return True
