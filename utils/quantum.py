import numpy as np
import matplotlib.pyplot as plt
from .math import matexp, normalize
from .plot import _colorize_complex

sigma_x = np.array([
    [0, 1],
    [1, 0]
])
sigma_y = np.array([
    [0, -1j],
    [1j,  0]
])
sigma_z = np.array([
    [1,  0],
    [0, -1]
])

try:
    from qiskit import Aer, transpile, assemble, execute
    from qiskit import QuantumCircuit
    from qiskit.quantum_info.operators import Operator

    # Other useful imports
    from qiskit import ClassicalRegister, QuantumRegister
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    from qiskit.circuit.library import PhaseEstimation

    def run(circuit, shots=2**11, showstate=True, showqubits=None, figsize=(16,4)):
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
            plotQ(state, showqubits, figsize=figsize)
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

    def get_unitary(circ, decimals=3):
        sim = execute(circ, Aer.get_backend('unitary_simulator')) # run the simulator
        return sim.result().get_unitary(circ, decimals=decimals)

    def get_pe_energies(circ):
        u = get_unitary(circ)
        eigvals, eigvecs = np.linalg.eig(u)
        energies = np.angle(eigvals)/(2*np.pi)
        return energies

except ModuleNotFoundError:
    print("Warning: qiskit not installed!")
    pass

def reverse_qubit_order(state):
    state = np.array(state)
    n = int(np.log2(len(state)))
    return state.reshape([2]*n).T.flatten()

def partial_trace(rho, retain_qubits=[0,1]):
    rho = np.array(rho)
    n = int(np.log2(rho.shape[0]))
    trace_out = set(range(n)) - set(retain_qubits)
    for qubit in trace_out:
        rho = _partial_trace(rho, subsystem_dims=[2]*n, subsystem_to_trace_out=qubit)
        n = int(np.log2(rho.shape[0]))
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

def plotQ(state, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None):
    def tobin(n, places):
        return ("{0:0" + str(places) + "b}").format(n)

    def plotcoeff(ax):
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

    def plotprobs(ax):
        toshow = {}
        cumsum = 0
        for idx in probs.argsort()[-20:][::-1]: # only look at 20 largest
            if cumsum > 0.96 or probs[idx] < 0.01:
                break
            toshow[tobin(idx, n)] = probs[idx]
            cumsum += probs[idx]
        toshow["rest"] = max(0,1-cumsum)
        ax.pie(toshow.values(), labels=toshow.keys(), autopct=lambda x: f"%.1f%%" % x)

    def plotrho(ax):
        rho = np.outer(state, state.conj())
        rho = partial_trace(rho, retain_qubits=showqubits)
        rho = _colorize_complex(rho)
        ax.imshow(rho)
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            ax.set_xticks(range(rho.shape[0]), basis)
            ax.set_yticks(range(rho.shape[0]), basis)
            ax.tick_params(axis="x", rotation=45)

    state = np.array(state)
    n = int(np.log2(len(state))) # nr of qubits
    probs = np.abs(state)**2

    # trace out unwanted qubits
    if showqubits is None:
        showqubits = range(n)
    else:
        # sanity checks
        if not hasattr(showqubits, '__len__'):
            showqubits = [showqubits]
        if len(showqubits) == 0:
            showqubits = range(n)
        elif max(showqubits) >= n:
            raise ValueError(f"No such qubit: %d" % max(showqubits))

        state = state.reshape(tuple([2]*n))
        probs = probs.reshape(tuple([2]*n))

        cur = 0
        for i in range(n):
            if i not in showqubits:
                state = np.sum(state, axis=cur)
                probs = np.sum(probs, axis=cur)
            else:
                cur += 1
        state = state.flatten()
        state = normalize(state) # renormalize
        n = int(np.log2(len(state))) # update n
        probs = probs.flatten()
        assert np.abs(np.sum(probs) - 1) < 1e-5, np.sum(probs)

    if showcoeff and showprobs and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,3, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotrho(axs[0])
        plotcoeff(axs[1])
        plotprobs(axs[2])
    elif showcoeff and showprobs:
        if figsize is None:
            figsize=(18,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0])
        plotprobs(axs[1])
    elif showcoeff and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0])
        plotrho(axs[1])
    elif showprobs and showrho:
        if figsize is None:
            figsize=(6,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        plotrho(axs[0])
        plotprobs(axs[1])
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if showcoeff:
            plotcoeff(ax)
        elif showprobs:
            plotprobs(ax)
        elif showrho:
            plotrho(ax)

    fig.tight_layout()
    plt.show()
