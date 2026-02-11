# qsvm_demo.py
# A demonstration of how to use QSVM for a classification task using qiskit-machine-learning.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.algorithms import QSVM
from qiskit_machine_learning.kernels import QuantumKernel

# 1. Generate some sample data (similar to a fraud detection problem)
# We'll use 2 features for simplicity
n_features = 2
X, y = make_classification(
    n_samples=40,
    n_features=n_features,
    n_informative=n_features,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Scale the data to be between 0 and 1, which is suitable for many feature maps
X = MinMaxScaler().fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 2. Set up the QSVM
# Define the feature map - this encodes the classical data into a quantum state
feature_map = ZFeatureMap(feature_dimension=n_features, reps=1)

# Set up the quantum instance - this defines the backend to run the simulation on
# We'll use the statevector_simulator for this demo
backend = BasicAer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(
    backend, shots=1024, seed_simulator=42, seed_transpiler=42
)

# Set up the quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

# 3. Instantiate and train the QSVM
qsvm = QSVM(quantum_kernel=quantum_kernel)

print("Training the QSVM...")
qsvm.fit(X_train, y_train)

# 4. Evaluate the QSVM
print("\nEvaluating the QSVM...")
qsvm_score = qsvm.score(X_test, y_test)
print(f"QSVM score on the test set: {qsvm_score:.4f}")

# You can also make predictions on new data
# predictions = qsvm.predict(new_data)
