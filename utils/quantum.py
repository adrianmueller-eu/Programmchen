import numpy as np
from .math import expm, normalize
from .plot import plotQ

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

def reverse_qubit_order(state):
    state = np.array(state)
    n = int(np.log2(len(state)))
    return state.reshape([2]*n).T.flatten()

try:
    from qiskit import Aer, transpile, assemble, execute
    from qiskit import QuantumCircuit
    from qiskit.quantum_info.operators import Operator

    def run(circuit, shots=2**11, showstate=True, showqubits=None, figsize=(16,4)):
        if showstate:
            simulator = Aer.get_backend("statevector_simulator")
        else:
            simulator = Aer.get_backend('aer_simulator')
        t_circuit = transpile(circuit, simulator)
        result = simulator.run(t_circuit, shots=shots).result()

        if showstate:
            state = np.array(result.get_statevector())
            plotQ(state, showqubits, figsize=figsize)
            return result, state
        else:
            return result

    class exp_i(QuantumCircuit):
        def __init__(self, H):
            self.H = H
            self.n = int(np.log2(len(self.H)))
            super().__init__(self.n)     # circuit on n qubits
            u = expm(1j*self.H)          # create unitary from hamiltonian
            self.all_qubits = list(range(self.n))
            self.unitary(Operator(u), self.all_qubits, label="exp^iH") # add unitary to circuit

        def power(self, k):
            q_k = QuantumCircuit(self.n)
            u = expm(1j*k*self.H)
            q_k.unitary(Operator(u), self.all_qubits, label=f"exp^i{k}H")
            return q_k

    def get_unitary(circ, decimals=3):
        sim = execute(circ, Aer.get_backend('unitary_simulator')) # run the simulator
        return sim.result().get_unitary(circ, decimals=decimals)

