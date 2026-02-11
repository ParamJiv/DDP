import numpy as np
import bosonic_qiskit
import qiskit
import sys # Import sys for printing to stderr

print("Script started.", file=sys.stdout) # To check if script begins execution

# 1. Setup Qumode and Classical Registers
# 2 qumodes, 1 qubit per qumode (allowing 0 or 1 photon per qumode)
num_qumodes = 2
num_qubits_per_qumode = 1 # To encode Fock states |0> and |1>
qmr = bosonic_qiskit.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode)
# Classical register (not strictly needed for statevector, but for completeness)
cr = qiskit.ClassicalRegister(qmr.num_qubits) 

# 2. Create CVCircuit
circuit = bosonic_qiskit.CVCircuit(qmr, cr)

# 3. Prepare an initial state: |1,0> (one photon in qumode 0, zero in qumode 1)
circuit.cv_fock(1, qmr[0]) # 1 photon in qumode 0

print("Initial state prepared: |1,0>", file=sys.stdout)
print("Applying a 50/50 Beamsplitter between qumode 0 and qumode 1", file=sys.stdout)

# 4. Apply a Beamsplitter gate
# Beamsplitter with theta = np.pi/4 (50/50 beamsplitter)
circuit.cv_bs(np.pi/4, qmr[0], qmr[1])

# 5. Simulate the circuit
try:
    state, _, _ = bosonic_qiskit.util.simulate(circuit)
    
    print("\nSimulation successful!", file=sys.stdout)
    print("Probabilities of Fock states after beamsplitter:", file=sys.stdout)
    
    state_probabilities = state.probabilities()
    
    # Manually decode the bitstring to Fock states
    for i, prob in enumerate(state_probabilities):
        if prob > 1e-6: # Only print significant probabilities
            bitstring = format(i, '0{}b'.format(qmr.num_qubits)) # Total 2 qubits: q1 q0
            
            # Decode for qumode 0 (qubit 0, least significant bit)
            fock_q0 = int(bitstring[1]) 
            
            # Decode for qumode 1 (qubit 1, most significant bit)
            fock_q1 = int(bitstring[0]) 
            
            print(f"  Fock state |{fock_q0},{fock_q1}>: {prob:.4f}", file=sys.stdout)
            
except Exception as e:
    print(f"Simulation failed: {e}", file=sys.stderr) # Print errors to stderr

print("Script finished.", file=sys.stdout) # To check if script finishes execution
