import numpy as np
import matplotlib.pyplot as plt
from .mathlib import matexp, normalize, is_hermitian, is_unitary
from .plot import colorize_complex

#############
### Gates ###
#############

def Rx(theta):
   return np.array([
              [np.cos(theta/2), -1j*np.sin(theta/2)],
              [-1j*np.sin(theta/2), np.cos(theta/2)]
          ], dtype=complex)

def Ry(theta):
   return np.array([
              [np.cos(theta/2), -np.sin(theta/2)],
              [np.sin(theta/2), np.cos(theta/2)]
          ], dtype=complex)

def Rz(theta):
   return np.array([
              [np.exp(-1j*theta/2), 0],
              [0, np.exp(1j*theta/2)]
          ], dtype=complex)

fs = lambda x: 1/np.sqrt(x)
f2 = fs(2)
I = np.eye(2)
X = np.array([ # 1j*Rx(np.pi)
    [0, 1],
    [1, 0]
], dtype=complex)
Y = np.array([ # 1j*Ry(np.pi)
    [0, -1j],
    [1j,  0]
], dtype=complex)
Z = np.array([ # 1j*Rz(np.pi)
    [1,  0],
    [0, -1]
], dtype=complex)
S = np.array([
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
], dtype=complex)
C_ = lambda A: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, A[0,0], A[0,1]],
    [0, 0, A[1,0], A[1,1]]
], dtype=complex)
CX = C_(X)
CNOT = CX
SWAP = np.array([ # CNOT @ r(reverse_qubit_order(CNOT)) @ CNOT
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

try:
    ##############
    ### Qiskit ###
    ##############

    from qiskit import Aer, transpile, assemble, execute
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.quantum_info.operators import Operator

    # Other useful imports
    from qiskit.quantum_info import random_statevector, Statevector
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    from qiskit.circuit.library import *

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

def ising_model(n_qubits, J, h=None, g=None, offset=0, kind='2d'):
    """
    Generates an Ising model with (optional) longitudinal and (optional) transverse couplings.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, at least 2, has to be a perfect square if kind='2d'.
    J : float or array
        Coupling strength. If a scalar, all couplings are set to this value.
        If a 2-element vector, all couplings are set to a random value in this range.
        If a matrix, this matrix is used as the coupling matrix. Uses only the upper triangular part.
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
    kind : {'1d', '2d'}, optional
        Whether the couplings are 1d or 2d.

    Returns
    -------
    H : str
        The Hamiltonian as a string, which can be parsed by parse_hamiltonian.
    """
    # parse arguments
    assert n_qubits > 2, "n_qubits must be greater than 2"
    if kind == '2d':
        # assert np.sqrt(n_qubits) % 1 == 0, "n_qubits must be a perfect square for kind='2d'"
        if hasattr(J, '__len__') and len(J) == 2:
            J = np.random.uniform(J[0], J[1], (n_qubits, n_qubits))
        # check if J is scalar or matrix
        assert np.isscalar(J) or J.shape == (n_qubits, n_qubits), "J must be a scalar, 2-element vector, or matrix of shape (n_qubits, n_qubits)"
        # use only upper triangular part
        if not np.isscalar(J):
            J = np.triu(J)
    elif kind == '1d':
        if hasattr(J, '__len__') and len(J) == 2:
            J = np.random.uniform(J[0], J[1], n_qubits)
        assert np.isscalar(J) or len(J) == n_qubits, "J must be a scalar, 2-element vector, or vector of length n_qubits"
    else:
        raise ValueError(f"Unknown kind {kind}")
    if h is not None:
        if hasattr(h, '__len__') and len(h) == 2:
            h = np.random.uniform(low=h[0], high=h[1], size=n_qubits)
        assert np.isscalar(h) or len(h) == n_qubits, "h must be a scalar, 2-element vector, or vector of length n_qubits"
    if g is not None:
        if hasattr(g, '__len__') and len(g) == 2:
            g = np.random.uniform(low=g[0], high=g[1], size=n_qubits)
        assert np.isscalar(g) or len(g) == n_qubits, "g must be a scalar, 2-element vector, or vector of length n_qubits"

    # generate the Hamiltonian
    H_str = ''
    # pairwise interactions
    if kind == '2d':
        if np.isscalar(J):
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    H_str += 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1) + ' + '
            H_str = str(J) + '*(' + H_str[:-3] + ') + '
        else:
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if J[i,j] != 0:
                        H_str += str(J[i,j]) + '*' + 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1) + ' + '
    elif kind == '1d':
        if np.isscalar(J):
            for i in range(n_qubits-1):
                H_str += 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            H_str += 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
            H_str = str(J) + '*(' + H_str[:-3] + ') + '
        else:
            for i in range(n_qubits-1):
                if J[i] != 0:
                    H_str += str(J[i]) + '*' + 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if J[n_qubits-1] != 0:
                H_str += str(J[n_qubits-1]) + '*' + 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
    else:
        raise ValueError(f"Unknown kind {kind}")
    # local longitudinal fields
    if np.any(h):
        if np.isscalar(h):
            H_str += str(h) + '*(' + ' + '.join(['I'*i + 'Z' + 'I'*(n_qubits-i-1) for i in range(n_qubits)]) + ') + '
        else:
            H_str += ' + '.join([str(h[i]) + '*' + 'I'*i + 'Z' + 'I'*(n_qubits-i-1) for i in range(n_qubits) if h[i] != 0]) + ' + '
    # local transverse fields
    if np.any(g):
        if np.isscalar(g):
            H_str += str(g) + '*(' + ' + '.join(['I'*i + 'X' + 'I'*(n_qubits-i-1) for i in range(n_qubits)]) + ') + '
        else:
            H_str += ' + '.join([str(g[i]) + '*' + 'I'*i + 'X' + 'I'*(n_qubits-i-1) for i in range(n_qubits) if g[i] != 0]) + ' + '
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

def tests_quantum_all():
    tests = [
        _test_get_H_energies_eq_get_pe_energies,
        _test_parse_hamiltonian1,
        _test_parse_hamiltonian2,
        _test_parse_hamiltonian3,
        _test_reverse_qubit_order1,
        _test_reverse_qubit_order2,
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
