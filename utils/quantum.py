import numpy as np
import matplotlib.pyplot as plt
from .mathlib import matexp, normalize
from .plot import colorize_complex

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

    def get_pe_energies(circ):
        u = get_unitary(circ)
        eigvals, eigvecs = np.linalg.eig(u)
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
    print("Warning: qiskit not installed!")
    pass

def reverse_qubit_order(state):
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
        n = int(np.log2(rho.shape[0]))
        trace_out -= 1 # rename the axes (only "higher" ones are left)
    return rho

# from https://github.com/cvxpy/cvxpy/issues/563
def _partial_trace(rho, subsystem_dims, subsystem_to_trace_out=0):
    dims_ = np.array(subsystem_dims)
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))
    reshaped_rho = np.moveaxis(reshaped_rho, subsystem_to_trace_out, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims_)+subsystem_to_trace_out-1, -1)
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)
    dims_untraced = np.delete(dims_, subsystem_to_trace_out)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape(rho_dim, rho_dim)

def state_trace(state, retain_qubits):
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
    n = int(np.log2(len(state))) # update n
    probs = probs.flatten()
    assert np.abs(np.sum(probs) - 1) < 1e-5, np.sum(probs)

    return state, probs

def plotQ(state, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None):
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
    real = np.random.random(2**n)
    imag = np.random.random(2**n)
    return normalize(real + 1j*imag)
