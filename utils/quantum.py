import numpy as np
import itertools
import matplotlib.pyplot as plt
from .mathlib import matexp, normalize, is_hermitian, is_unitary
from .plot import colorize_complex

#############
### Gates ###
#############

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
#T = np.array([ # avoid overriding T = True
#    [1,  0],
#    [0,  np.sqrt(1j)]
#], dtype=complex)
H = 1/np.sqrt(2) * np.array([
    [1,  1],
    [1, -1]
], dtype=complex) # f2*(X + Z)

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
CX = C_(X)
CNOT = CX
SWAP = np.array([ # 0.5*(XX + YY + ZZ + II), CNOT @ r(reverse_qubit_order(CNOT)) @ CNOT
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
II = I_(2)
Toffoli = C_(C_(X))


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

    def run(circuit, shots=2**0, showstate=True, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=(16,4)):
        if shots > 10:
            tc = time_complexity(circuit)
            print("TC: %d, expected running time: %.3fs" % (tc, tc * 0.01))
        if showstate:
            simulator = Aer.get_backend("statevector_simulator")
        else:
            simulator = Aer.get_backend('aer_simulator')
        t_circuit = transpile(circuit, simulator)
        result = simulator.run(t_circuit, shots=shots).result()

        if showstate:
            state = np.array(result.get_statevector())
            # qiskit outputs the qubits in the reverse order
            state = reverse_qubit_order(state)
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
    elif state.shape[0] == state.shape[1]:
        vals, vecs = np.linalg.eig(state)
        vecs = np.array([reverse_qubit_order(vecs[:,i]) for i in range(2**n)])
        return vecs.T @ np.diag(vals) @ vecs

def partial_trace(rho, retain_qubits=[0,1]):
    """Trace out all qubits not specified in `retain_qubits`."""
    rho = np.array(rho)
    if len(rho.shape) == 1:
        rho = np.outer(rho, rho.conj())
    assert rho.shape[0] == rho.shape[1], f"Can't trace a matrix of size {rho.shape}"

    n = int(np.log2(rho.shape[0]))
    if type(retain_qubits) == "int":
        retain_qubits = [retain_qubits]
    trace_out = np.array(sorted(set(range(n)) - set(retain_qubits)))
    for qubit in trace_out:
        rho = _partial_trace(rho, subsystem_dims=[2]*n, subsystem_to_trace_out=qubit)
        n -= 1         # one qubit less
        trace_out -= 1 # rename the axes (only "higher" ones are left)
    return rho

def _partial_trace(rho, subsystem_dims, subsystem_to_trace_out=0):
    """Traces out `subsystem_to_trace_out`-th qubit from the given density matrix `rho`. Edited version of the one found in https://github.com/cvxpy/cvxpy/issues/563"""
    dims_ = np.array(subsystem_dims)
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))
    reshaped_rho = np.moveaxis(reshaped_rho, subsystem_to_trace_out, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims_)+subsystem_to_trace_out-1, -1)
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)
    dims_untraced = np.delete(dims_, subsystem_to_trace_out)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape(rho_dim, rho_dim)

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

###################
### Hamiltonian ###
###################

def parse_hamiltonian(hamiltonian):
    """Parse a string representation of a Hamiltonian into a matrix representation.

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
    def s(gate):
        return globals()[gate]

    # Remove whitespace
    hamiltonian = hamiltonian.replace(" ", "")
    # replace - with +-, except before e
    hamiltonian = hamiltonian \
                    .replace("-", "+-") \
                    .replace("e+-", "e-") \
                    .replace("(+-", "(-")

    # Find parts in parentheses
    part = ""
    parts = []
    depth = 0
    for c in hamiltonian:
        if c == "(":
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
            parts.append(part)
            part = ""

    # Replace parts in parentheses with a placeholder
    for i, part in enumerate(parts):
        hamiltonian = hamiltonian.replace(part, f"part{i}", 1)
        # Calculate the part recursively
        parts[i] = parse_hamiltonian(part[1:-1])

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

    H = np.zeros((2**n, 2**n), dtype=complex)
    for chunk in chunks:

        # print(chunk, hamiltonian)
        chunk_matrix = None
        if chunk == "":
            continue
        # Parse the weight of the chunk
        try:
            if "*" in chunk:
                weight = complex(chunk.split("*")[0])
                chunk = chunk.split("*")[1]
            elif "part" in chunk and chunk[0] != "p":
                weight = complex(chunk.split("part")[0])
                chunk = "part" + chunk.split("part")[1]
            else:
                weight = complex(chunk)
                chunk_matrix = np.eye(2**n)
        except ValueError:
            if chunk[0] == "-":
                weight = -1
                chunk = chunk[1:]
            elif chunk[0] == "+":
                weight = 1
                chunk = chunk[1:]
            else:
                weight = 1
        # If the chunk is a part, add it to the Hamiltonian
        if chunk_matrix is not None:
            pass
        elif chunk.startswith("part"):
            chunk_matrix = parts[int(chunk.split("part")[1])]
        else:
            if len(chunk) != n:
                raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")

            # Get the matrix representation of the chunk
            chunk_matrix = s(chunk[0])
            for gate in chunk[1:]:
                chunk_matrix = np.kron(chunk_matrix, s(gate))

        # Add the chunk to the Hamiltonian
        # print("chunk", weight, chunk, hamiltonian, parts)
        H += weight * chunk_matrix

    assert np.allclose(H, H.conj().T), "Hamiltonian must be Hermitian"

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
    if kind == '1d':
        assert type(n_qubits) == int or len(n_qubits) == 1, f"For kind={kind}, n_qubits must be an integer or tuple of length 1, but is {n_qubits}"
        # convert to int if tuple (has attr __len__)
        if hasattr(n_qubits, '__len__'):
            n_qubits = n_qubits[0]
        couplings = (n_qubits if circular and n_qubits > 2 else n_qubits-1,)
    elif kind == '2d':
        if type(n_qubits) == int or len(n_qubits) == 1:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 2, but is {n_qubits}")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == '3d':
        if type(n_qubits) == int or len(n_qubits) == 2:
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
        assert J.shape == couplings, f"For kind={kind}, J must be a scalar, 2-element vector, or matrix of shape {couplings}, but is {J.shape}"
    elif isinstance(J, dict) and kind != 'full':
        raise ValueError(f"For kind={kind}, J must not be a dict!")

    if h is not None:
        if n_total_qubits != 2 and hasattr(h, '__len__') and len(h) == 2:
            h = np.random.uniform(low=h[0], high=h[1], size=n_qubits)
        elif not np.isscalar(h):
            h = np.array(h)
        assert np.isscalar(h) or h.shape == (n_qubits,), f"h must be a scalar, 2-element vector, or vector of length {n_qubits}, but is {h.shape if not np.isscalar(h) else h}"
    if g is not None:
        if n_total_qubits != 2 and hasattr(g, '__len__') and len(g) == 2:
            g = np.random.uniform(low=g[0], high=g[1], size=n_qubits)
        elif not np.isscalar(g):
            g = np.array(g)
        assert np.isscalar(g) or g.shape == (n_qubits,), f"g must be a scalar, 2-element vector, or vector of length {n_qubits}, but is {g.shape if not np.isscalar(g) else g}"

    # generate the Hamiltonian
    H_str = ''
    # pairwise interactions
    if kind == '1d':
        if np.isscalar(J):
            for i in range(n_qubits-1):
                H_str += 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if circular:
                H_str += 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
        else:
            for i in range(n_qubits-1):
                if J[i] != 0:
                    H_str += str(J[i]) + '*' + 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if circular and J[n_qubits-1] != 0:
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
                for membership in itertools.combinations(range(n_qubits), i):
                    H_str += ''.join(['Z' if j in membership else 'I' for j in range(n_qubits)]) + ' + '
        else: # J is a dict of tuples of qubit indices to interaction strengths
            for membership, strength in J.items():
                if strength == 0:
                    continue
                H_str += str(strength) + '*' + ''.join(['Z' if j in membership else 'I' for j in range(n_qubits)]) + ' + '
    else:
        raise ValueError(f"Unknown kind {kind}")

    if np.isscalar(J):
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


#############
### Tests ###
#############

def test_quantum_all():
    tests = [
        _test_get_H_energies_eq_get_pe_energies,
        _test_parse_hamiltonian1,
        _test_parse_hamiltonian2,
        _test_parse_hamiltonian3,
        _test_reverse_qubit_order1,
        _test_reverse_qubit_order2,
        _test_ising_model
    ]

    for test in tests:
        print("Running", test.__name__, "... ", end="")
        if test():
            print("Test succeed!")
        else:
            print("ERROR!")
            break

def _test_get_H_energies_eq_get_pe_energies():
    n_qubits = np.random.randint(1,10)
    n_terms = np.random.randint(1,100)
    H = random_hamiltonian(n_qubits, n_terms, scaling=False)
    H = parse_hamiltonian(H)

    A = np.sort(get_pe_energies(exp_i(H)))
    B = get_H_energies(H)
    return np.allclose(A, B)

def _test_parse_hamiltonian1():
    H = parse_hamiltonian('0.5*(II + ZI - ZX + IX)')
    return np.allclose(H, CNOT)

def _test_parse_hamiltonian2():
    H = parse_hamiltonian('0.5*(XX + YY + ZZ + II)')
    return np.allclose(H, SWAP)

def _test_parse_hamiltonian3():
    H = parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    return np.allclose(np.sum(H), 2)

def _test_reverse_qubit_order1():
    H = parse_hamiltonian('IIIXX')
    H_rev = parse_hamiltonian('XXIII')

    H_rev2 = reverse_qubit_order(H)
    return np.allclose(H_rev, H_rev2)

def _test_reverse_qubit_order2():
    psi = np.kron([1,1], [0,1])
    psi_rev = np.kron([0,1], [1,1])

    psi_rev2 = reverse_qubit_order(psi)
    return np.allclose(psi_rev, psi_rev2)

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

