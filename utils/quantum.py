import numpy as np
from itertools import combinations, product
from functools import reduce
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .mathlib import normalize, matexp, matlog, is_psd, is_hermitian
from .plot import colorize_complex

#################
### Unitaries ###
#################

fs = lambda x: 1/np.sqrt(x)
f2 = fs(2)
I_ = lambda n: np.eye(2**n)
I = I_(1)
X = np.array([ # 1j*Rx(pi)
    [0, 1],
    [1, 0]
], dtype=complex)
Y = np.array([ # 1j*Ry(pi)
    [0, -1j],
    [1j,  0]
], dtype=complex)
Z = np.array([ # 1j*Rz(pi)
    [1,  0],
    [0, -1]
], dtype=complex)
S = np.array([ # np.sqrt(Z)
    [1,  0],
    [0, 1j]
], dtype=complex)
T_gate = np.array([ # avoid overriding T = True
    [1,  0],
    [0,  np.sqrt(1j)]
], dtype=complex)
Had = 1/np.sqrt(2) * np.array([
    [1,  1],
    [1, -1]
], dtype=complex) # f2*(X + Z) = 1j*f2*(Rx(pi) + Rz(pi))

def R_(gate, theta):
   return matexp(-1j*gate*theta/2)

Rx = lambda theta: R_(X, theta)
Ry = lambda theta: R_(Y, theta)
Rz = lambda theta: R_(Z, theta)

def C_(A):
    if not hasattr(A, 'shape'):
        A = np.array(A, dtype=complex)
    n = int(np.log2(A.shape[0]))
    return np.kron([[1,0],[0,0]], I_(n)) + np.kron([[0,0],[0,1]], A)
CNOT = CX = C_(X) # 0.5*(II + ZI - ZX + IX)
Toffoli = C_(C_(X))
SWAP = np.array([ # 0.5*(XX + YY + ZZ + II), CNOT @ r(reverse_qubit_order(CNOT)) @ CNOT
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
iSWAP = np.array([ # 0.5*(1j*(XX + YY) + ZZ + II), R_(XX+YY, -pi/2)
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

def parse_unitary(unitary):
    """Parse a string representation of a unitary into its matrix representation. The result is guaranteed to be unitary.

    Example:
    >>> parse_unitary('CX @ XC @ CX') # SWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
           [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]])
    >>> parse_unitary('SS @ HI @ CX @ XC @ IH') # iSWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
           [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    """
    def s(chunk):
        # think of XCXC, which applies X on the first and fifth qubit, controlled on the second and forth qubit
        # select the first "C", then recursively call s on the part before and after (if they exist), and combine them afterwards
        chunk_matrix = np.array([1]) # initialize with a stub
        for i, c in enumerate(chunk):
            if c == "C":
                n_before = i
                n_after = len(chunk) - i - 1
                # if it's the first C, then there is no part before
                if n_before == 0:
                    return np.kron([[1,0],[0,0]], I_(n_after)) + np.kron([[0,0],[0,1]], s(chunk[i+1:]))
                # if it's the last C, then there is no part after
                elif n_after == 0:
                    return np.kron(I_(n_before), [[1,0],[0,0]]) + np.kron(chunk_matrix, [[0,0],[0,1]])
                # if it's in the middle, then there is a part before and after
                else:
                    return np.kron(I_(n_before), np.kron([[1,0],[0,0]], I_(n_after))) + np.kron(chunk_matrix, np.kron([[0,0],[0,1]], s(chunk[i+1:])))
            # N is negative control, so it's the same as C, but with the roles of 0 and 1 reversed
            elif c == "N":
                n_before = i
                n_after = len(chunk) - i - 1
                # if it's the first N, then there is no part before
                if n_before == 0:
                    return np.kron([[0,0],[0,1]], I_(n_after)) + np.kron([[1,0],[0,0]], s(chunk[i+1:]))
                # if it's the last N, then there is no part after
                elif n_after == 0:
                    return np.kron(I_(n_before), [[0,0],[0,1]]) + np.kron(chunk_matrix, [[1,0],[0,0]])
                # if it's in the middle, then there is a part before and after
                else:
                    return np.kron(I_(n_before), np.kron([[0,0],[0,1]], I_(n_after))) + np.kron(chunk_matrix, np.kron([[1,0],[0,0]], s(chunk[i+1:])))
            # if there is no C, then it's just a single gate
            elif c == "T":
                 gate = T_gate
            elif c == "H":
                 gate = Had
            else:
                 gate = globals()[chunk[i]]
            chunk_matrix = np.kron(chunk_matrix, gate)
        return chunk_matrix

    # Remove whitespace
    unitary = unitary.replace(" ", "")

    # Parse the unitary
    chunks = unitary.split("@")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # Use the first chunk to determine the number of qubits
    n = len(chunks[0])

    U = np.eye(2**n, dtype=complex)
    for chunk in chunks:
        # print(chunk, unitary)
        chunk_matrix = None
        if chunk == "":
            continue
        if len(chunk) != n:
            raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")

        # Get the matrix representation of the chunk
        chunk_matrix = s(chunk)

        # Update the unitary
        # print("chunk", chunk, unitary, chunk_matrix)
        U = U @ chunk_matrix

    assert np.allclose(U @ U.conj().T, np.eye(2**n)), f"Result is not unitary: {U, U @ U.conj().T}"

    return U

XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
II = I_(2)


try:
    ##############
    ### Qiskit ###
    ##############

    from qiskit import Aer, transpile, assemble, execute
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.quantum_info.operators import Operator

    # Other useful imports
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_histogram
    #from qiskit.circuit.library import *

    def run(circuit, shots=2**0, generate_state=True, plot=True, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=(16,4)):
        if shots > 10:
            tc = time_complexity(circuit)
            print("TC: %d, expected running time: %.3fs" % (tc, tc * 0.01))
        if generate_state:
            simulator = Aer.get_backend("statevector_simulator")
        else:
            simulator = Aer.get_backend('aer_simulator')
        t_circuit = transpile(circuit, simulator)
        result = simulator.run(t_circuit, shots=shots).result()

        if generate_state:
            state = np.array(result.get_statevector())
            # qiskit outputs the qubits in the reverse order
            state = reverse_qubit_order(state)
            if plot:
                plotQ(state, showqubits=showqubits, showcoeff=showcoeff, showprobs=showprobs, showrho=showrho, figsize=figsize)
            return result, state
        else:
            return result

    class exp_i(QuantumCircuit):
        def __init__(self, H):
            self.H = H
            self.n = int(np.log2(len(self.H)))
            super().__init__(self.n)       # circuit on n qubits
            u = matexp(1j*self.H)          # create unitary from hamiltonian
            self.all_qubits = list(range(self.n))
            self.unitary(Operator(u), self.all_qubits, label="exp^iH") # add unitary to circuit

        def power(self, k):
            q_k = QuantumCircuit(self.n)
            u = matexp(1j*k*self.H)
            q_k.unitary(Operator(u), self.all_qubits, label=f"exp^i{k}H")
            return q_k

    def get_unitary(circ, decimals=42):
        sim = execute(circ, Aer.get_backend('unitary_simulator')) # run the simulator
        return sim.result().get_unitary(circ, decimals=decimals)

    def get_pe_energies(U):
        if isinstance(U, QuantumCircuit):
            U = get_unitary(U)
        eigvals, eigvecs = np.linalg.eig(U)
        energies = np.angle(eigvals)/(2*np.pi)
        return energies

    def show_eigenvecs(circ, showrho=False):
        u = get_unitary(circ)
        eigvals, eigvecs = np.linalg.eig(u)
        print(np.round(eigvecs, 3))
        for i in range(eigvecs.shape[1]):
            plotQ(eigvecs[:,i], figsize=(12,2), showrho=showrho)
        return eigvecs

    def time_complexity(qc, decompose_iterations=4, isTranspiled=False):
        if not isTranspiled:
            simulator = Aer.get_backend('aer_simulator')
            t_circuit = transpile(qc, simulator)
        else:
            t_circuit = qc
        for _ in range(decompose_iterations):
            t_circuit = t_circuit.decompose()
        return len(t_circuit._data)


except ModuleNotFoundError:
    print("Warning: qiskit not installed! Use `pip install qiskit`.")
    pass

#############
### State ###
#############

def reverse_qubit_order(state):
    """So the last will be first, and the first will be last. Works for both, state vectors and density matrices."""
    state = np.array(state)
    n = int(np.log2(len(state)))

    # if vector, just reshape
    if len(state.shape) == 1 or state.shape[0] != state.shape[1]:
        return state.reshape([2]*n).T.flatten()
    # if matrix, reverse qubit order in eigenvectors
    # TODO: This doesn't work for mixed states!
    elif state.shape[0] == state.shape[1]:
        vals, vecs = np.linalg.eig(state)
        vecs = np.array([reverse_qubit_order(vecs[:,i]) for i in range(2**n)])
        return vecs.T @ np.diag(vals) @ vecs

def partial_trace(rho, retain_qubits):
    """Trace out all qubits not specified in `retain_qubits`."""
    rho = np.array(rho)
    n = int(np.log2(rho.shape[0])) # number of qubits

    # pre-process retain_qubits
    if isinstance(retain_qubits, int):
        retain_qubits = [retain_qubits]
    dim_r = 2**len(retain_qubits)

    # get qubits to trace out
    trace_out = np.array(sorted(set(range(n)) - set(retain_qubits)))
    # ignore all qubits >= n
    trace_out = trace_out[trace_out < n]

    # if rho is a state vector
    if len(rho.shape) == 1:
        st  = rho.reshape([2]*n)
        rho = np.tensordot(st, st.conj(), axes=(trace_out,trace_out))
    # if trace out all qubits, just return the normal trace
    elif len(trace_out) == n:
        return np.trace(rho).reshape(1,1) # dim_r is not necessarily 1 here (if some in `retain_qubits` are >= n)
    else:
        assert rho.shape[0] == rho.shape[1], f"Can't trace a non-square matrix {rho.shape}"

        rho = rho.reshape([2]*(2*n))
        for qubit in trace_out:
            rho = np.trace(rho, axis1=qubit, axis2=qubit+n)
            n -= 1         # one qubit less
            trace_out -= 1 # rename the axes (only "higher" ones are left)

    return rho.reshape(dim_r, dim_r)

def state_trace(state, retain_qubits):
    """This is a pervert version of the partial trace, but for state vectors. I'm not sure about the physical meaning of its output, but it was at times helpful to visualize and interpret subsystems, especially when the density matrix was out of reach (or better: out of memory)."""
    state = np.array(state)
    state[np.isnan(state)] = 0
    n = int(np.log2(len(state))) # nr of qubits

    # sanity checks
    if not hasattr(retain_qubits, '__len__'):
        retain_qubits = [retain_qubits]
    if len(retain_qubits) == 0:
        retain_qubits = range(n)
    elif n == len(retain_qubits):
        return state, np.abs(state)**2
    elif max(retain_qubits) >= n:
        raise ValueError(f"No such qubit: %d" % max(retain_qubits))

    state = state.reshape(tuple([2]*n))
    probs = np.abs(state)**2

    cur = 0
    for i in range(n):
        if i not in retain_qubits:
            state = np.sum(state, axis=cur)
            probs = np.sum(probs, axis=cur)
        else:
            cur += 1

    state = state.flatten()
    state = normalize(state) # renormalize

    probs = probs.flatten()
    assert np.abs(np.sum(probs) - 1) < 1e-5, np.sum(probs) # sanity check

    return state, probs

def plotQ(state, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None):
    """My best attempt so far to visualize a state vector. Control with `showqubits` which subsystem you're interested in (`None` will show the whole state). `showcoeff` utilitzes `state_trace`, `showprobs` shows a pie chart of the probabilities when measured in the standard basis, and `showrho` gives a plt.imshow view on the corresponding density matrix."""

    def tobin(n, places):
        return ("{0:0" + str(places) + "b}").format(n)

    def plotcoeff(ax, state):
        n = int(np.log2(len(state))) # nr of qubits
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            #plot(basis, state, ".", figsize=(10,3))
            ax.scatter(basis, state.real, label="real")
            ax.scatter(basis, state.imag, label="imag")
            ax.tick_params(axis="x", rotation=45)
        elif n < 9:
            ax.scatter(range(2**n), state.real, label="real")
            ax.scatter(range(2**n), state.imag, label="imag")
        else:
            ax.plot(range(2**n), state.real, label="real")
            ax.plot(range(2**n), state.imag, label="imag")

        #from matplotlib.ticker import StrMethodFormatter
        #ax.xaxis.set_major_formatter(StrMethodFormatter("{x:0"+str(n)+"b}"))
        ax.legend()
        ax.grid()

    def plotprobs(ax, state):
        n = int(np.log2(len(state))) # nr of qubits
        toshow = {}
        cumsum = 0
        for idx in probs.argsort()[-20:][::-1]: # only look at 20 largest
            if cumsum > 0.96 or probs[idx] < 0.01:
                break
            toshow[tobin(idx, n)] = probs[idx]
            cumsum += probs[idx]
        if np.abs(1-cumsum) > 1e-15:
            toshow["rest"] = max(0,1-cumsum)
        ax.pie(toshow.values(), labels=toshow.keys(), autopct=lambda x: f"%.1f%%" % x)

    def plotrho(ax, rho):
        n = int(np.log2(len(state))) # nr of qubits
        rho = colorize_complex(rho)
        ax.imshow(rho)
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            ax.set_xticks(range(rho.shape[0]), basis)
            ax.set_yticks(range(rho.shape[0]), basis)
            ax.tick_params(axis="x", rotation=45)

    state = np.array(state)

    # trace out unwanted qubits
    if showqubits is None:
        n = int(np.log2(len(state))) # nr of qubits
        showqubits = range(n)

    if showrho:
        import psutil
        memory_requirement = (4*len(state))**2
        #print(memory_requirement / 1024**2, "MB") # rho.nbytes
        if memory_requirement > psutil.virtual_memory().available:
            raise ValueError(f"Too high memory requirement (%.1f GB) to calulate the density matrix!" % (memory_requirement / 1024**3))
        rho = np.outer(state, state.conj())
        rho = partial_trace(rho, retain_qubits=showqubits)
    state, probs = state_trace(state, showqubits)

    if showcoeff and showprobs and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,3, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotrho(axs[0], rho)
        plotcoeff(axs[1], state)
        plotprobs(axs[2], state)
    elif showcoeff and showprobs:
        if figsize is None:
            figsize=(18,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0], state)
        plotprobs(axs[1], state)
    elif showcoeff and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0], state)
        plotrho(axs[1], rho)
    elif showprobs and showrho:
        if figsize is None:
            figsize=(6,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        plotrho(axs[0], rho)
        plotprobs(axs[1], state)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if showcoeff:
            plotcoeff(ax, state)
        elif showprobs:
            plotprobs(ax, state)
        elif showrho:
            plotrho(ax, rho)

    fig.tight_layout()
    plt.show()

def random_state(n=1):
    """Generate a random state vector ($2^{n+1}-2$ degrees of freedom). Normalized and without global phase."""
    real = np.random.random(2**n)
    imag = np.random.random(2**n)
    return normalize(real + 1j*imag)

def random_density_matrix(n=1, pure=True):
    """Generate a random density matrix ($2^{n+1}-1$ degrees of freedom). Normalized and without global phase."""
    if pure:
        state = random_state(n)
        return np.outer(state, state.conj())
    else:
        probs = normalize(np.random.random(2**n), p=1)
        res = np.zeros((2**n, 2**n), dtype=complex)
        for p in probs:
            state = random_state(n)
            res += p * np.outer(state, state.conj())
        return res

def state(specification):
    """Convert a string or dictionary of strings and weights to a state vector. The string can be a binary number or a combination of binary numbers and weights. The weights will be normalized to 1."""
    # if a string is given, convert it to a dictionary
    if type(specification) == str:
        # remove whitespace
        specification = specification.replace(" ", "")
        specification_dict = dict()
        specification = specification.split("+")
        # re-merge parts with parentheses, e.g. (0.5+1j)*100
        for i, s in enumerate(specification):
            if s[0] == "(" and s[-1] != ")":
                specification[i] += "+" + specification[i+1]
                specification.pop(i+1)
        # parse the weights
        for s in specification:
            if "*" in s:
                weight, state = s.split("*")
                specification_dict[state] = weight
            else:
                specification_dict[s] = 1
        specification = specification_dict
        # convert the weights to floats
        for key in specification:
            specification[key] = complex(specification[key])
    # convert the dictionary to a state vector
    n = len(list(specification.keys())[0])
    state = np.zeros(2**n, dtype=complex)
    for key in specification:
        state[int(key, 2)] = specification[key]
    return normalize(state)

def op(specification1, specification2=None):
    if type(specification1) == str:
        s1 = state(specification1)
    elif hasattr(specification1, '__len__'):
        s1 = np.array(specification1)
    else:
        raise ValueError(f"Unknown specification: {specification1}")
    if specification2 is None:
        s2 = s1
    elif type(specification2) == str:
        s2 = state(specification2)
    elif hasattr(specification2, '__len__'):
        s2 = np.array(specification2)
    else:
        raise ValueError(f"Unknown specification: {specification2}")
    return np.outer(s1, s2.conj())

def probs(state):
    """Calculate the probabilities of measuring a state vector in the standard basis."""
    return np.abs(state)**2

def von_Neumann_entropy(state):
    """Calculate the von Neumann entropy of a state vector."""
    state = np.array(state)
    S = -np.trace(state @ matlog(state)/np.log(2))
    assert np.allclose(S.imag, 0), f"WTF: Entropy is not real: {S}"
    return np.max(S.real, 0)  # fix rounding errors

def entanglement_entropy(state, subsystem_qubits):
    """Calculate the entanglement entropy of a state vector with respect to the given subsystem."""
    return von_Neumann_entropy(partial_trace(state, subsystem_qubits))

def is_dm(rho):
    """Check if matrix `rho` is a density matrix."""
    rho = np.array(rho)
    return is_psd(rho) and np.allclose(np.trace(rho), 1)

###################
### Hamiltonian ###
###################

matmap_np, matmap_sp = None, None

def parse_hamiltonian(hamiltonian, sparse=False, scaling=1, buffer=None, max_buffer_n=0, dtype=complex):
    """Parse a string representation of a Hamiltonian into a matrix representation. The result is guaranteed to be Hermitian.

    Parameters:
        hamiltonian (str): The Hamiltonian to parse.
        sparse (bool): Whether to use sparse matrices (csr_matrix) or dense matrices (numpy.array).
        scaling (float): A constant factor to scale the Hamiltonian by.
        buffer (dict): A dictionary to store calculated chunks in. If `None`, it defaults to the global `matmap_np` (or `matmap_sp` if `sparse == True`). Give `buffer={}` and leave `max_buffer_n == 0` (default) to disable the buffer.
        max_buffer_n (int): The maximum length (number of qubits) for new chunks to store in the buffer (default: 0). If `0`, no new chunks will be stored in the buffer.

    Returns:
        numpy.ndarray | scipy.sparse.csr_matrix: The matrix representation of the Hamiltonian.

    Example:
    >>> parse_hamiltonian('0.5*(XX + YY + ZZ + II)') # SWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
           [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  2.+0.j -2.+0.j  0.+0.j]
           [ 0.+0.j -2.+0.j  2.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('0.5*(II + ZI - ZX + IX)') # CNOT

    """
    kron = sp.kron if sparse else np.kron

    # Initialize the matrix map
    global matmap_np, matmap_sp
    if matmap_np is None or matmap_sp is None:
        # numpy versions
        matmap_np = {
            "H": np.array(Had, dtype=dtype),
            "X": np.array(X, dtype=dtype),
            "Y": np.array(Y, dtype=dtype),
            "Z": np.array(Z, dtype=dtype),
            "I": np.array(I, dtype=dtype),
            "II": np.eye(2**2, dtype=dtype),
            "ZZ": np.array(np.kron(Z, Z), dtype=dtype),
            "IX": np.array(np.kron(I, X), dtype=dtype),
            "XI": np.array(np.kron(X, I), dtype=dtype),
            "III": np.eye(2**3, dtype=dtype),
            "IIII": np.eye(2**4, dtype=dtype),
            "IIIII": np.eye(2**5, dtype=dtype),
            "IIIIII": np.eye(2**6, dtype=dtype),
            "IIIIIII": np.eye(2**7, dtype=dtype),
            "IIIIIIII": np.eye(2**8, dtype=dtype),
            "IIIIIIIII": np.eye(2**9, dtype=dtype),
            "IIIIIIIIII": np.eye(2**10, dtype=dtype),
        }

        # sparse versions
        matmap_sp = {k: sp.csr_array(v) for k, v in matmap_np.items()}

    matmap = matmap_sp if sparse else matmap_np

    # only use buffer if pre-computed chunks are available or if new chunks are allowed to be stored
    use_buffer = buffer is None or len(buffer) > 0 or max_buffer_n > 0
    if use_buffer and buffer is None:
        buffer = matmap

    def calculate_chunk_matrix(chunk, sparse=False, scaling=1):
        if use_buffer:
            if chunk in buffer:
                return buffer[chunk] if scaling == 1 else scaling * buffer[chunk]
            if len(chunk) == 1:
                return matmap[chunk[0]] if scaling == 1 else scaling * matmap[chunk[0]]
            # Check if a part of the chunk has already been calculated
            for i in range(len(chunk)-1, 1, -1):
                for j in range(len(chunk)-i+1):
                    subchunk = chunk[j:j+i]
                    if subchunk in buffer:
                        # If so, calculate the rest of the chunk recursively
                        parts = [chunk[:j], subchunk, chunk[j+i:]]
                        # remove empty chunks
                        parts = [c for c in parts if c != ""]
                        # See where to apply the scaling
                        shortest = min(parts, key=len)
                        # Calculate each in tomultiply recursively
                        for i, c in enumerate(parts):
                            if c == subchunk:
                                parts[i] = buffer[c]
                            parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=scaling if c == shortest else 1)
                        return reduce(kron, parts)

        # Calculate the chunk matrix gate by gate
        if use_buffer and len(chunk) <= max_buffer_n:
            gates = [matmap[gate] for gate in chunk]
            chunk_matrix = reduce(kron, gates)
            buffer[chunk] = chunk_matrix
            if scaling != 1:
                chunk_matrix = scaling * chunk_matrix
        else:
            gates = [scaling * matmap[chunk[0]]] + [matmap[gate] for gate in chunk[1:]]
            chunk_matrix = reduce(kron, gates)

        return chunk_matrix

    # Remove whitespace
    hamiltonian = hamiltonian.replace(" ", "")
    # replace - with +-, except before e
    hamiltonian = hamiltonian \
                    .replace("-", "+-") \
                    .replace("e+-", "e-") \
                    .replace("(+-", "(-")

    # print("parse_hamiltonian: Pre-processed Hamiltonian:", hamiltonian)

    # Find parts in parentheses
    part = ""
    parts = []
    depth = 0
    current_part_weight = ""
    for i, c in enumerate(hamiltonian):
        if c == "(":
            if depth == 0:
                # for top-level parts search backwards for the weight
                weight = ""
                for j in range(i-1, -1, -1):
                    if hamiltonian[j] in ["("]:
                        break
                    weight += hamiltonian[j]
                    if hamiltonian[j] in ["+", "-"]:
                        break
                weight = weight[::-1]
                if weight != "":
                    current_part_weight = weight
            depth += 1
        elif c == ")":
            depth -= 1
        if depth > 0:
            part += c
        if depth == 0 and c == ")":
            part += c
            # reject parts with complex numbers
            try:
                assert complex(part).imag == 0, "Complex coefficients lead to non-hermitian matrices"
            except ValueError:
                pass
            parts.append((current_part_weight, part))
            part = ""
            current_part_weight = ""

    # print("Parts found:", parts)

    # Replace parts in parentheses with a placeholder
    for i, (weight, part) in enumerate(parts):
        hamiltonian = hamiltonian.replace(weight+part, f"part{i}", 1)
        if weight in ["", "+", "-"]:
            weight += "1"
        if weight[-1] == "*":
            weight = weight[:-1]
        # Calculate the part recursively
        parts[i] = parse_hamiltonian(part[1:-1], sparse=sparse, scaling=scaling * float(weight), buffer=buffer, max_buffer_n=max_buffer_n, dtype=dtype)

    # print("Parts replaced:", parts)

    # Parse the rest of the Hamiltonian
    chunks = hamiltonian.split("+")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # If parts are present, use them to determine the number of qubits
    if parts:
        n = int(np.log2(parts[0].shape[0]))
    else: # Use the first chunk to determine the number of qubits
        first_chunk = chunks[0]
        if first_chunk[0] in ["-", "+"]:
            first_chunk = first_chunk[1:]
        try:
            n = len(first_chunk.split("*")[1])
        except IndexError:
            n = len(first_chunk)

    if sparse:
        H = sp.csr_array((2**n, 2**n), dtype=dtype)
    else:
        if n > 10:
            raise ValueError(f"Using a dense matrix for a {n}-qubit Hamiltonian is not recommended. Use sparse=True.")
        H = np.zeros((2**n, 2**n), dtype=dtype)

    for chunk in chunks:

        # print("Processing chunk:", chunk)
        chunk_matrix = None
        if chunk == "":
            continue
        # Parse the weight of the chunk
        try:
            if len(chunk) == n:
                weight = 1
            elif "*" in chunk:
                weight = float(chunk.split("*")[0])
                chunk = chunk.split("*")[1]
            elif "part" in chunk:
                weight = 1
            elif len(chunk) == n+1 and chunk[0] in ["-", "+"]:
                weight = float(chunk[0] + "1")
                chunk = chunk[1:]
            else:
                weight = float(chunk)
                chunk_matrix = weight * np.eye(2**n, dtype=dtype)
        except ValueError:
                raise ValueError(f"Invalid chunk for size {n}: {chunk}")

        # If the chunk is a part, add it to the Hamiltonian
        if chunk_matrix is not None:
            pass
        elif chunk.startswith("part"):
            chunk_matrix = parts[int(chunk.split("part")[1])]
        else:
            if len(chunk) != n:
                raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")

            chunk_matrix = calculate_chunk_matrix(chunk, sparse=sparse, scaling = scaling * weight)

        # Add the chunk to the Hamiltonian
        # print("Adding chunk", weight, chunk, "to hamiltonian", scaling, hamiltonian)
        # print(type(H), H.dtype, type(chunk_matrix), chunk_matrix.dtype)
        if len(chunks) == 1:
            H = chunk_matrix
        else:
            H += chunk_matrix

    if sparse:
        assert np.allclose(H.data, H.conj().T.data), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H.data}"
    else:
        assert np.allclose(H, H.conj().T), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H}"

    return H

def random_hamiltonian(n_qubits, n_terms, offset=0, gates='XYZI', scaling=True):
    """Generate `n_terms` combinations out of `n_qubits` gates drawn from `gates`. If `scaling=True`, the scaling factor is `1/n_terms`."""
    # generate a list of random terms
    combs = [''.join(np.random.choice(list(gates), n_qubits)) for _ in range(n_terms)]
    H_str = ' + '.join(combs)
    if scaling:
        H_str = str(1/len(combs)) + '*(' + H_str + ')'
    if offset != 0:
        H_str += ' + ' + str(offset)
    return H_str

def ising_model(n_qubits, J, h=None, g=None, offset=0, kind='1d', circular=False):
    """
    Generates an Ising model with (optional) longitudinal and (optional) transverse couplings.

    Parameters
    ----------
    n_qubits : int or tuple
        Number of qubits, at least 2. For `kind='2d'` or `kind='3d'`, give a tuple of 2 or 3 integers, respectively.
    J : float, array, or dict
        Coupling strength. If a scalar, all couplings are set to this value.
        If a 2-element vector but `n_qubit > 2` (or tuple), all couplings are set to a random value in this range.
        If a matrix, this matrix is used as the coupling matrix.
        For `kind='pairwise'`, `kind='2d'`, or `kind='3d'`, `J` is read as an incidence matrix, where the rows and columns correspond to the qubits and the values are the coupling strengths. Only the upper triangular part of the matrix is used.
        For `kind='full'`, specify a dictionary with keys being tuples of qubit indices and values being the corresponding coupling strength.
    h : float or array, optional
        Longitudinal field strength. If a scalar, all fields are set to this value.
        If a 2-element vector, all fields are set to a random value in this range.
        If a vector of size `n_qubits`, its elements specify the individual strengths of the longitudinal field.
    g : float or array, optional
        Transverse field strength. If a scalar, all couplings are set to this value.
        If a 2-element vector, all couplings are set to a random value in this range.
        If a vector of size `n_qubits`, its elements specify the individual strengths of the transverse field.
    offset : float, optional
        Offset of the Hamiltonian.
    kind : {'1d', '2d', '3d', 'pairwise', 'full'}, optional
        Whether the couplings are along a string (`1d`), on a 2d-lattice (`2d`), 3d-lattice (`3d`), fully connected graph (`pairwise`), or specify the desired multi-particle interactions.
    circular : bool, optional
        Whether the couplings are circular (i.e. the outermost qubits are coupled to each other). Only applies to `kind='1d'`, `kind='2d'`, and `kind='3d'`.

    Returns
    -------
    H : str
        The Hamiltonian as a string, which can be parsed by parse_hamiltonian.
    """
    # generate the coupling shape
    n_total_qubits = np.prod(n_qubits)
    assert n_total_qubits - int(n_total_qubits) == 0, "n_qubits must be an integer or a tuple of integers"
    if kind == '1d':
        assert np.isscalar(n_qubits) or len(n_qubits) == 1, f"For kind={kind}, n_qubits must be an integer or tuple of length 1, but is {n_qubits}"
        # convert to int if tuple (has attr __len__)
        if hasattr(n_qubits, '__len__'):
            n_qubits = n_qubits[0]
        couplings = (n_qubits if circular and n_qubits > 2 else n_qubits-1,)
    elif kind == '2d':
        if np.isscalar(n_qubits) or len(n_qubits) == 1:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 2, but is {n_qubits}")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == '3d':
        if np.isscalar(n_qubits) or len(n_qubits) == 2:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 3, but is {n_qubits}")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == 'pairwise':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits}"
        couplings = (n_qubits, n_qubits)
    elif kind == 'full':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits}"
        couplings = (2**n_qubits,)
    else:
        raise ValueError(f"Unknown kind {kind}")

    # if J is not scalar or dict, it must be either the array of the couplings or the limits of the random range
    if not (np.isscalar(J) or isinstance(J, dict)):
        J = np.array(J)
        if J.shape == (2,):
            J = np.random.uniform(J[0], J[1], couplings)
        if kind == '1d' and J.shape == (n_qubits, n_qubits):
            # get the offset k=1 diagonal (n_qubits-1 elements)
            idxs = np.where(np.eye(n_qubits, k=1))
            if circular:
                # add the edge element
                idxs = (np.append(idxs[0], 0), np.append(idxs[1], n_qubits-1))
            J = J[idxs]
        assert J.shape == couplings, f"For kind={kind}, J must be a scalar, 2-element vector, or matrix of shape {couplings}, but is {J.shape}"
    elif isinstance(J, dict) and kind != 'full':
        raise ValueError(f"For kind={kind}, J must not be a dict!")

    if h is not None:
        if n_total_qubits != 2 and hasattr(h, '__len__') and len(h) == 2:
            h = np.random.uniform(low=h[0], high=h[1], size=n_total_qubits)
        elif not np.isscalar(h):
            h = np.array(h)
        assert np.isscalar(h) or h.shape == (n_total_qubits,), f"h must be a scalar, 2-element vector, or vector of shape {(n_total_qubits,)}, but is {h.shape if not np.isscalar(h) else h}"
    if g is not None:
        if n_total_qubits != 2 and hasattr(g, '__len__') and len(g) == 2:
            g = np.random.uniform(low=g[0], high=g[1], size=n_total_qubits)
        elif not np.isscalar(g):
            g = np.array(g)
        assert np.isscalar(g) or g.shape == (n_total_qubits,), f"g must be a scalar, 2-element vector, or vector of shape {(n_total_qubits,)}, but is {g.shape if not np.isscalar(g) else g}"

    # generate the Hamiltonian
    H_str = ''
    # pairwise interactions
    if kind == '1d':
        if np.isscalar(J):
            for i in range(n_qubits-1):
                H_str += 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if circular and n_qubits > 2:
                H_str += 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
        else:
            for i in range(n_qubits-1):
                if J[i] != 0:
                    H_str += str(J[i]) + '*' + 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if circular and n_qubits > 2 and J[n_qubits-1] != 0:
                H_str += str(J[n_qubits-1]) + '*' + 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
    elif kind == '2d':
        for i in range(n_qubits[0]):
            for j in range(n_qubits[1]):
                # find all 2d neighbors, but avoid double counting
                neighbors = []
                if i > 0:
                    neighbors.append((i-1, j))
                if i < n_qubits[0]-1:
                    neighbors.append((i+1, j))
                if j > 0:
                    neighbors.append((i, j-1))
                if j < n_qubits[1]-1:
                    neighbors.append((i, j+1))
                if circular:
                    if i == n_qubits[0]-1 and n_qubits[0] > 2:
                        neighbors.append((0, j))
                    if j == n_qubits[1]-1 and n_qubits[1] > 2:
                        neighbors.append((i, 0))
                # add interactions
                index_node = i*n_qubits[1] + j
                for neighbor in neighbors:
                    # 1. lower row
                    # 2. same row, but further to the right or row circular (= first column and j is last column)
                    # 3. same column, but column circular (= first row and i is last row)
                    if neighbor[0] > i \
                        or (neighbor[0] == i and (neighbor[1] > j or (j == n_qubits[1]-1 and neighbor[1] == 0 and n_qubits[1] > 2))) \
                        or (neighbor[1] == j and i == n_qubits[0]-1 and neighbor[0] == 0 and n_qubits[0] > 2):
                        index_neighbor = neighbor[0]*n_qubits[1] + neighbor[1]
                        first_index = min(index_node, index_neighbor)
                        second_index = max(index_node, index_neighbor)
                        if not np.isscalar(J):
                            if J[first_index, second_index] == 0:
                                continue
                            H_str += str(J[first_index, second_index]) + '*'
                        H_str += 'I'*first_index + 'Z' + 'I'*(second_index-first_index-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]-second_index-1) + ' + '
    elif kind == '3d':
        for i in range(n_qubits[0]):
            for j in range(n_qubits[1]):
                for k in range(n_qubits[2]):
                    # find all 3d neighbors, but avoid double counting
                    neighbors = []
                    if i > 0:
                        neighbors.append((i-1, j, k))
                    if i < n_qubits[0]-1:
                        neighbors.append((i+1, j, k))
                    if j > 0:
                        neighbors.append((i, j-1, k))
                    if j < n_qubits[1]-1:
                        neighbors.append((i, j+1, k))
                    if k > 0:
                        neighbors.append((i, j, k-1))
                    if k < n_qubits[2]-1:
                        neighbors.append((i, j, k+1))
                    if circular:
                        if i == n_qubits[0]-1 and n_qubits[0] > 2:
                            neighbors.append((0, j, k))
                        if j == n_qubits[1]-1 and n_qubits[1] > 2:
                            neighbors.append((i, 0, k))
                        if k == n_qubits[2]-1 and n_qubits[2] > 2:
                            neighbors.append((i, j, 0))
                    # add interactions
                    index_node = i*n_qubits[1]*n_qubits[2] + j*n_qubits[2] + k
                    for neighbor in neighbors:
                        # 1. lower row
                        # 2. same row, but
                            # a. same layer, but further to the right or row circular (= first column and j is last column)
                            # b. same column, but further behind or layer circular (= first layer and k is last layer)
                        # 3. same column and same layer, but column circular (= first row and i is last row)
                        if neighbor[0] > i \
                            or (neighbor[0] == i and (\
                                (neighbor[2] == k and (neighbor[1] > j or (j == n_qubits[1]-1 and neighbor[1] == 0 and n_qubits[1] > 2))) \
                                or (neighbor[1] == j and (neighbor[2] > k or (k == n_qubits[2]-1 and neighbor[2] == 0 and n_qubits[2] > 2))) \
                            )) \
                            or (neighbor[1] == j and neighbor[2] == k and i == n_qubits[0]-1 and neighbor[0] == 0 and n_qubits[0] > 2):
                            index_neighbor = neighbor[0]*n_qubits[1]*n_qubits[2] + neighbor[1]*n_qubits[2] + neighbor[2]
                            first_index = min(index_node, index_neighbor)
                            second_index = max(index_node, index_neighbor)
                            if not np.isscalar(J):
                                if J[first_index, second_index] == 0:
                                    continue
                                H_str += str(J[first_index, second_index]) + '*' 
                            H_str += 'I'*first_index + 'Z' + 'I'*(second_index-first_index-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]*n_qubits[2]-second_index-1) + ' + '
    elif kind == 'pairwise':
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if not np.isscalar(J):
                    if J[i,j] == 0:
                        continue
                    H_str += str(J[i,j]) + '*'
                H_str += 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1) + ' + '
    elif kind == 'full':
        if np.isscalar(J):
            if n_qubits > 20:
                raise ValueError("Printing out all interactions for n_qubits > 20 is not recommended. Please use a dict instead.")
            for i in range(2, n_qubits+1):
                for membership in combinations(range(n_qubits), i):
                    H_str += ''.join(['Z' if j in membership else 'I' for j in range(n_qubits)]) + ' + '
        else: # J is a dict of tuples of qubit indices to interaction strengths
            for membership, strength in J.items():
                if strength == 0:
                    continue
                H_str += str(strength) + '*' + ''.join(['Z' if j in membership else 'I' for j in range(n_qubits)]) + ' + '
    else:
        raise ValueError(f"Unknown kind {kind}")

    if np.isscalar(J) and J != 0 and n_total_qubits > 1:
        H_str = str(J) + '*(' + H_str[:-3] + ') + '

    # local longitudinal fields
    if np.any(h):
        if np.isscalar(h):
            H_str += str(h) + '*(' + ' + '.join(['I'*i + 'Z' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits)]) + ') + '
        else:
            H_str += ' + '.join([str(h[i]) + '*' + 'I'*i + 'Z' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits) if h[i] != 0]) + ' + '
    # local transverse fields
    if np.any(g):
        if np.isscalar(g):
            H_str += str(g) + '*(' + ' + '.join(['I'*i + 'X' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits)]) + ') + '
        else:
            H_str += ' + '.join([str(g[i]) + '*' + 'I'*i + 'X' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits) if g[i] != 0]) + ' + '
    # offset
    if np.any(offset):
        H_str += str(offset)
    else:
        H_str = H_str[:-3]
    return H_str

def get_H_energies(H, expi=True):
    """Returns the energies of the given hamiltonian `H`. For `expi=True` (default) it gives the same result as `get_pe_energies(exp_i(H))` (up to sorting) and for `expi=False` it returns the eigenvalues of `H`."""
    if type(H) == str:
        H = parse_hamiltonian(H)
    energies = np.linalg.eigvalsh(H)
    if expi:
        energies = (energies % (2*np.pi))/(2*np.pi)
        energies[energies > 0.5] -= 1
        energies = np.sort(energies)
    return energies

def pauli_basis(n, kind='np', normalize=False):
    """ Generate the pauli basis of hermitian 2**n x 2**n matrices. This basis is orthonormal and, except for the identity, traceless.

    E.g. for n = 2, the basis is [II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    Parameters
        n (int): Number of qubits
        kind (str): 'np' for numpy arrays (default), 'sp' for scipy sparse matrices, or 'str' for strings
        normalize (bool): Whether to normalize the basis elements (default False)

    Returns
        list[ np.ndarray | scipy.sparse.csr_matrix | str ]: The pauli basis
    """
    def reduce_norm(f, l, normalize):
        if normalize:
            # apply norm np.sqrt(2**n) to the first element, and reduce the rest
            first = l[0]/np.sqrt(2**n)
            if len(l) == 1:
                return first
            rest = reduce(f, l[1:])
            return f(first, rest)
        else:
            return reduce(f, l)

    if kind == 'np':
        return [reduce_norm(np.kron, i, normalize) for i in product([I,X,Y,Z], repeat=n)]
    elif kind == 'sp':
        basis = [sp.csr_array(b) for b in [I,X,Y,Z]]
        return [reduce_norm(sp.kron, i, normalize) for i in product(basis, repeat=n)]
    elif kind == 'str':
        norm_str = f"{1/np.sqrt(2**n)}*" if normalize else ""
        return [norm_str + ''.join(i) for i in product(['I', 'X', 'Y', 'Z'], repeat=n)]
    else:
        raise ValueError(f"Unknown kind: {kind}")

#############
### Tests ###
#############

def test_quantum_all():
    tests = [
        _test_get_H_energies_eq_get_pe_energies,
        _test_parse_hamiltonian,
        _test_reverse_qubit_order,
        _test_partial_trace,
        _test_von_Neumann_entropy,
        _test_entanglement_entropy,
        _test_ising_model,
        _test_pauli_basis
    ]

    for test in tests:
        print("Running", test.__name__, "... ", end="", flush=True)
        if test():
            print("Test succeed!", flush=True)
        else:
            print("ERROR!")
            break

def _test_get_H_energies_eq_get_pe_energies():
    n_qubits = np.random.randint(1, 5)
    n_terms = np.random.randint(1, 100)
    H = random_hamiltonian(n_qubits, n_terms, scaling=False)
    H = parse_hamiltonian(H)

    A = np.sort(get_pe_energies(exp_i(H)))
    B = get_H_energies(H)
    return np.allclose(A, B)

def _test_parse_hamiltonian():
    H = parse_hamiltonian('0.5*(II + ZI - ZX + IX)')
    assert np.allclose(H, CNOT)

    H = parse_hamiltonian('0.5*(XX + YY + ZZ + II)')
    assert np.allclose(H, SWAP)

    H = parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    assert np.allclose(np.sum(H), 2)

    H = parse_hamiltonian('0.2*(-0.5*(3*XX + 4*YY) + 1*II)')
    assert np.allclose(np.sum(H), -.4)

    return True

def _test_reverse_qubit_order():
    # known 3-qubit matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    psi_rev = np.kron(np.kron([1,-1], [0,1]), [1,1])
    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev, psi_rev2)

    # same as above, but with n random qubits
    n = 10
    psis = [random_state(1) for _ in range(n)]
    psi = psis[0]
    for i in range(1,n):
        psi = np.kron(psi, psis[i])
    psi_rev = psis[-1]
    for i in range(1,n):
        psi_rev = np.kron(psi_rev, psis[-i-1])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev, psi_rev2)

    # general hamiltonian
    H = parse_hamiltonian('IIIXX')
    H_rev = parse_hamiltonian('XXIII')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev, H_rev2)

    # pure density matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    rho = np.outer(psi, psi)
    psi_rev = np.kron(np.kron([1,-1], [0,1]), [1,1])
    rho_rev = np.outer(psi_rev, psi_rev)
    rho_rev2 = reverse_qubit_order(rho)
    assert np.allclose(rho_rev, rho_rev2)

    # TODO: This test fails
    # # draw n times 2 random 1-qubit states and a probability distribution over all n pairs
    # n = 10
    # psis = [[random_density_matrix(1) for _ in range(2)] for _ in range(n)]
    # p = normalize(np.random.rand(n), p=1)
    # # compute the average state
    # psi = np.zeros((2**2, 2**2), dtype=complex)
    # for i in range(n):
    #     psi += p[i]*np.kron(psis[i][0], psis[i][1])
    # # compute the average state with reversed qubit order
    # psi_rev = np.zeros((2**2, 2**2), dtype=complex)
    # for i in range(n):
    #     psi_rev += p[i]*np.kron(psis[i][1], psis[i][0])

    # psi_rev2 = reverse_qubit_order(psi)
    # assert np.allclose(psi_rev, psi_rev2), f"psi_rev = {psi_rev}\npsi_rev2 = {psi_rev2}"

    return True

def _test_partial_trace():
    # known 4x4 matrix
    rho = np.arange(16).reshape(4,4)
    rhoA_expected = np.array([[ 5, 9], [21, 25]])
    rhoA_actual   = partial_trace(rho, 0)
    assert np.allclose(rhoA_expected, rhoA_actual), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    # two separable density matrices
    rhoA = random_density_matrix(2)
    rhoB = random_density_matrix(3)
    rho = np.kron(rhoA, rhoB)
    rhoA_expected = rhoA
    rhoA_actual   = partial_trace(rho, [0,1])
    assert np.allclose(rhoA_expected, rhoA_actual), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    # two separable state vectors
    psiA = random_state(2)
    psiB = random_state(3)
    psi = np.kron(psiA, psiB)
    psiA_expected = np.outer(psiA, psiA.conj())
    psiA_actual   = partial_trace(psi, [0,1])
    assert np.allclose(psiA_expected, psiA_actual), f"psi_expected = {psiA_expected}\npsi_actual = {psiA_actual}"

    # total trace
    st = random_state(3)
    st_tr = partial_trace(st, [])
    assert np.allclose(np.array([[1]]), st_tr), f"st_tr = {st_tr} ≠ 1"
    rho = random_density_matrix(3)
    rho_tr = partial_trace(rho, [])
    assert np.allclose(np.array([[1]]), rho_tr), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    # retain all qubits
    st = random_state(3)
    st_tr = partial_trace(st, [0,1,2])
    st_expected = np.outer(st, st.conj())
    assert st_expected.shape == st_tr.shape, f"st_expected.shape = {st_expected.shape} ≠ st_tr.shape = {st_tr.shape}"
    assert np.allclose(st_expected, st_tr), f"st_expected = {st_expected} ≠ st_tr = {st_tr}"
    rho = random_density_matrix(2)
    rho_tr = partial_trace(rho, [0,1])
    assert rho.shape == rho_tr.shape, f"rho.shape = {rho.shape} ≠ rho_tr.shape = {rho_tr.shape}"
    assert np.allclose(rho, rho_tr), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    return True

def _test_von_Neumann_entropy():
    rho = random_density_matrix(2, pure=True)
    S = von_Neumann_entropy(rho)
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

    rho = np.eye(2)/2
    S = von_Neumann_entropy(rho)
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

    return True

def _test_entanglement_entropy():
    # Bell state |00> + |11> should for the first qubit have entropy 1
    rho = 1/2*np.outer(np.array([1,0,0,1]), np.array([1,0,0,1]))
    S = entanglement_entropy(rho, [0])
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

    # Two separable systems should for the first system have entropy 0
    rhoA = random_density_matrix(2)
    rhoB = random_density_matrix(3)
    rho = np.kron(rhoA, rhoB)
    S = entanglement_entropy(rho, [0,1])
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

    return True

def _test_ising_model():
    # 1d
    H_str = ising_model(5, J=1.5, h=0, g=0, offset=0, kind='1d', circular=False)
    expected = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model(5, J=1.5, h=1.1, g=0.5, offset=0.5, kind='1d', circular=True)
    expected = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ + ZIIIZ) + 1.1*(ZIIII + IZIII + IIZII + IIIZI + IIIIZ) + 0.5*(XIIII + IXIII + IIXII + IIIXI + IIIIX) + 0.5"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model(3, J=[0.6,0.7,0.8], h=[0.1,0.2,0.7], g=[0.6,0.1,1.5], offset=0.5, kind='1d', circular=True)
    expected = "0.6*ZZI + 0.7*IZZ + 0.8*ZIZ + 0.1*ZII + 0.2*IZI + 0.7*IIZ + 0.6*XII + 0.1*IXI + 1.5*IIX + 0.5"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model(3, J=[0,1], h=[1,2], g=[2,5], offset=0.5, kind='1d', circular=True)
    # random, but count terms in H_str instead
    n_terms = len(H_str.split('+'))
    assert n_terms == 10, f"n_terms = {n_terms}\nexpected = 10"

    # 2d
    H_str = ising_model((2,2), J=1.5, h=0, g=0, offset=0, kind='2d', circular=False)
    expected = "1.5*(ZIZI + ZZII + IZIZ + IIZZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model((3,3), J=1.5, h=1.1, g=0.5, offset=0.5, kind='2d', circular=True)
    expected = "1.5*(ZIIZIIIII + ZZIIIIIII + IZIIZIIII + IZZIIIIII + IIZIIZIII + ZIZIIIIII + IIIZIIZII + IIIZZIIII + IIIIZIIZI + IIIIZZIII + IIIIIZIIZ + IIIZIZIII + IIIIIIZZI + ZIIIIIZII + IIIIIIIZZ + IZIIIIIZI + IIZIIIIIZ + IIIIIIZIZ) + 1.1*(ZIIIIIIII + IZIIIIIII + IIZIIIIII + IIIZIIIII + IIIIZIIII + IIIIIZIII + IIIIIIZII + IIIIIIIZI + IIIIIIIIZ) + 0.5*(XIIIIIIII + IXIIIIIII + IIXIIIIII + IIIXIIIII + IIIIXIIII + IIIIIXIII + IIIIIIXII + IIIIIIIXI + IIIIIIIIX) + 0.5"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    # 3d
    H_str = ising_model((2,2,3), kind='3d', J=1.8, h=0, g=0, offset=0, circular=False)
    expected = "1.8*(ZIIIIIZIIIII + ZIIZIIIIIIII + ZZIIIIIIIIII + IZIIIIIZIIII + IZIIZIIIIIII + IZZIIIIIIIII + IIZIIIIIZIII + IIZIIZIIIIII + IIIZIIIIIZII + IIIZZIIIIIII + IIIIZIIIIIZI + IIIIZZIIIIII + IIIIIZIIIIIZ + IIIIIIZIIZII + IIIIIIZZIIII + IIIIIIIZIIZI + IIIIIIIZZIII + IIIIIIIIZIIZ + IIIIIIIIIZZI + IIIIIIIIIIZZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model((2,2,3), kind='3d', J=1.2, h=1.5, g=2, offset=0, circular=True)
    expected = "1.2*(ZIIIIIZIIIII + ZIIZIIIIIIII + ZZIIIIIIIIII + IZIIIIIZIIII + IZIIZIIIIIII + IZZIIIIIIIII + IIZIIIIIZIII + IIZIIZIIIIII + ZIZIIIIIIIII + IIIZIIIIIZII + IIIZZIIIIIII + IIIIZIIIIIZI + IIIIZZIIIIII + IIIIIZIIIIIZ + IIIZIZIIIIII + IIIIIIZIIZII + IIIIIIZZIIII + IIIIIIIZIIZI + IIIIIIIZZIII + IIIIIIIIZIIZ + IIIIIIZIZIII + IIIIIIIIIZZI + IIIIIIIIIIZZ + IIIIIIIIIZIZ) + 1.5*(ZIIIIIIIIIII + IZIIIIIIIIII + IIZIIIIIIIII + IIIZIIIIIIII + IIIIZIIIIIII + IIIIIZIIIIII + IIIIIIZIIIII + IIIIIIIZIIII + IIIIIIIIZIII + IIIIIIIIIZII + IIIIIIIIIIZI + IIIIIIIIIIIZ) + 2*(XIIIIIIIIIII + IXIIIIIIIIII + IIXIIIIIIIII + IIIXIIIIIIII + IIIIXIIIIIII + IIIIIXIIIIII + IIIIIIXIIIII + IIIIIIIXIIII + IIIIIIIIXIII + IIIIIIIIIXII + IIIIIIIIIIXI + IIIIIIIIIIIX)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model((3,3,3), kind='3d', J=1.5, h=0, g=0, offset=0, circular=True)
    expected = "1.5*(ZIIIIIIIIZIIIIIIIIIIIIIIIII + ZIIZIIIIIIIIIIIIIIIIIIIIIII + ZZIIIIIIIIIIIIIIIIIIIIIIIII + IZIIIIIIIIZIIIIIIIIIIIIIIII + IZIIZIIIIIIIIIIIIIIIIIIIIII + IZZIIIIIIIIIIIIIIIIIIIIIIII + IIZIIIIIIIIZIIIIIIIIIIIIIII + IIZIIZIIIIIIIIIIIIIIIIIIIII + ZIZIIIIIIIIIIIIIIIIIIIIIIII + IIIZIIIIIIIIZIIIIIIIIIIIIII + IIIZIIZIIIIIIIIIIIIIIIIIIII + IIIZZIIIIIIIIIIIIIIIIIIIIII + IIIIZIIIIIIIIZIIIIIIIIIIIII + IIIIZIIZIIIIIIIIIIIIIIIIIII + IIIIZZIIIIIIIIIIIIIIIIIIIII + IIIIIZIIIIIIIIZIIIIIIIIIIII + IIIIIZIIZIIIIIIIIIIIIIIIIII + IIIZIZIIIIIIIIIIIIIIIIIIIII + IIIIIIZIIIIIIIIZIIIIIIIIIII + IIIIIIZZIIIIIIIIIIIIIIIIIII + ZIIIIIZIIIIIIIIIIIIIIIIIIII + IIIIIIIZIIIIIIIIZIIIIIIIIII + IIIIIIIZZIIIIIIIIIIIIIIIIII + IZIIIIIZIIIIIIIIIIIIIIIIIII + IIIIIIIIZIIIIIIIIZIIIIIIIII + IIZIIIIIZIIIIIIIIIIIIIIIIII + IIIIIIZIZIIIIIIIIIIIIIIIIII + IIIIIIIIIZIIIIIIIIZIIIIIIII + IIIIIIIIIZIIZIIIIIIIIIIIIII + IIIIIIIIIZZIIIIIIIIIIIIIIII + IIIIIIIIIIZIIIIIIIIZIIIIIII + IIIIIIIIIIZIIZIIIIIIIIIIIII + IIIIIIIIIIZZIIIIIIIIIIIIIII + IIIIIIIIIIIZIIIIIIIIZIIIIII + IIIIIIIIIIIZIIZIIIIIIIIIIII + IIIIIIIIIZIZIIIIIIIIIIIIIII + IIIIIIIIIIIIZIIIIIIIIZIIIII + IIIIIIIIIIIIZIIZIIIIIIIIIII + IIIIIIIIIIIIZZIIIIIIIIIIIII + IIIIIIIIIIIIIZIIIIIIIIZIIII + IIIIIIIIIIIIIZIIZIIIIIIIIII + IIIIIIIIIIIIIZZIIIIIIIIIIII + IIIIIIIIIIIIIIZIIIIIIIIZIII + IIIIIIIIIIIIIIZIIZIIIIIIIII + IIIIIIIIIIIIZIZIIIIIIIIIIII + IIIIIIIIIIIIIIIZIIIIIIIIZII + IIIIIIIIIIIIIIIZZIIIIIIIIII + IIIIIIIIIZIIIIIZIIIIIIIIIII + IIIIIIIIIIIIIIIIZIIIIIIIIZI + IIIIIIIIIIIIIIIIZZIIIIIIIII + IIIIIIIIIIZIIIIIZIIIIIIIIII + IIIIIIIIIIIIIIIIIZIIIIIIIIZ + IIIIIIIIIIIZIIIIIZIIIIIIIII + IIIIIIIIIIIIIIIZIZIIIIIIIII + IIIIIIIIIIIIIIIIIIZIIZIIIII + IIIIIIIIIIIIIIIIIIZZIIIIIII + ZIIIIIIIIIIIIIIIIIZIIIIIIII + IIIIIIIIIIIIIIIIIIIZIIZIIII + IIIIIIIIIIIIIIIIIIIZZIIIIII + IZIIIIIIIIIIIIIIIIIZIIIIIII + IIIIIIIIIIIIIIIIIIIIZIIZIII + IIZIIIIIIIIIIIIIIIIIZIIIIII + IIIIIIIIIIIIIIIIIIZIZIIIIII + IIIIIIIIIIIIIIIIIIIIIZIIZII + IIIIIIIIIIIIIIIIIIIIIZZIIII + IIIZIIIIIIIIIIIIIIIIIZIIIII + IIIIIIIIIIIIIIIIIIIIIIZIIZI + IIIIIIIIIIIIIIIIIIIIIIZZIII + IIIIZIIIIIIIIIIIIIIIIIZIIII + IIIIIIIIIIIIIIIIIIIIIIIZIIZ + IIIIIZIIIIIIIIIIIIIIIIIZIII + IIIIIIIIIIIIIIIIIIIIIZIZIII + IIIIIIIIIIIIIIIIIIIIIIIIZZI + IIIIIIZIIIIIIIIIIIIIIIIIZII + IIIIIIIIIIIIIIIIIIZIIIIIZII + IIIIIIIIIIIIIIIIIIIIIIIIIZZ + IIIIIIIZIIIIIIIIIIIIIIIIIZI + IIIIIIIIIIIIIIIIIIIZIIIIIZI + IIIIIIIIZIIIIIIIIIIIIIIIIIZ + IIIIIIIIIIIIIIIIIIIIZIIIIIZ + IIIIIIIIIIIIIIIIIIIIIIIIZIZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    # pairwise
    H_str = ising_model(4, J=-.5, h=.4, g=.7, offset=1, kind='pairwise')
    expected = "-0.5*(ZZII + ZIZI + ZIIZ + IZZI + IZIZ + IIZZ) + 0.4*(ZIII + IZII + IIZI + IIIZ) + 0.7*(XIII + IXII + IIXI + IIIX) + 1"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    # full
    H_str = ising_model(3, J=1.5, h=.4, g=.7, offset=1, kind='full')
    expected = "1.5*(ZZI + ZIZ + IZZ + ZZZ) + 0.4*(ZII + IZI + IIZ) + 0.7*(XII + IXI + IIX) + 1"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising_model(3, kind='full', J={(0,1): 2, (0,1,2): 3, (1,2):0}, h=1.35)
    expected = "2*ZZI + 3*ZZZ + 1.35*(ZII + IZI + IIZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    J_dict = {
        (0,1): 1.5,
        (0,2): 2,
        (1,2): 0.5,
        (0,1,2): 3,
        (0,1,2,3): 0.5
    }
    H_str = ising_model(4, J=J_dict, h=.3, g=.5, offset=1.2, kind='full')
    expected = "1.5*ZZII + 2*ZIZI + 0.5*IZZI + 3*ZZZI + 0.5*ZZZZ + 0.3*(ZIII + IZII + IIZI + IIIZ) + 0.5*(XIII + IXII + IIXI + IIIX) + 1.2"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    return True

def _test_pauli_basis():
    n = np.random.randint(1,4)
    pauli_n = pauli_basis(n)

    # check the number of generators
    n_expected = 2**(2*n)
    assert len(pauli_n) == n_expected, f"Number of generators is {len(pauli_n)}, but should be {n_expected}!"

    # check if no two generators are the same
    for i, (A,B) in enumerate(combinations(pauli_n,2)):
        assert not np.allclose(A, B), f"Pair {i} is not different!"

    # check if all generators except of the identity are traceless
    assert np.allclose(pauli_n[0], np.eye(2**n)), "First generator is not the identity!"
    for i, A in enumerate(pauli_n[1:]):
        assert np.isclose(np.trace(A), 0), f"Generator {i} is not traceless!"

    # check if all generators are Hermitian
    for i, A in enumerate(pauli_n):
        assert is_hermitian(A), f"Generator {i} is not Hermitian!"

    # check if all generators are orthogonal
    for i, (A,B) in enumerate(combinations(pauli_n,2)):
        assert np.allclose(np.trace(A.conj().T @ B), 0), f"Pair {i} is not orthogonal!"

    # check normalization
    pauli_n_norm = pauli_basis(n, kind='np', normalize=True)
    for i, A in enumerate(pauli_n_norm):
        assert np.isclose(np.linalg.norm(A), 1), f"Generator {i} does not have norm 1!"

    # check string representation
    pauli_n_str = pauli_basis(n, kind='str')
    assert len(pauli_n) == len(pauli_n_str), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_str)):
        assert np.allclose(A, parse_hamiltonian(B)), f"Generator {i} is not the same!"

    # check sparse representation
    pauli_n_sp = pauli_basis(n, kind='sp')
    assert len(pauli_n) == len(pauli_n_sp), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_sp)):
        assert np.allclose(A, B.todense()), f"Generator {i} is not the same!"

    return True
