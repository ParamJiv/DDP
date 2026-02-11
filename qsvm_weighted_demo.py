# qsvm_weighted_demo.py
# A demonstration of how to use a quantum kernel with a weighted SVM for imbalanced classification.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel

# 1. Generate imbalanced sample data
n_features = 2
X, y = make_classification(
    n_samples=100,
    n_features=n_features,
    n_informative=n_features,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1 (anomalies)
    random_state=42,
)

# Scale the data
X = MinMaxScaler().fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 2. Set up the Quantum Kernel (same as before)
feature_map = ZFeatureMap(feature_dimension=n_features, reps=1)
backend = BasicAer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(
    backend, shots=1024, seed_simulator=42, seed_transpiler=42
)
# The .evaluate method returns a function that computes the kernel matrix
quantum_kernel_function = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance).evaluate

# 3. Use the quantum kernel with a classical SVM that supports class weights
# We can now use the 'class_weight' parameter from sklearn's SVC
# Option 1: 'balanced' - automatically adjusts weights
# svc = SVC(kernel=quantum_kernel_function, class_weight='balanced')

# Option 2: Manually specify weights. Give more weight to class 1 (anomalies)
class_weights = {0: 1, 1: 10} 
svc = SVC(kernel=quantum_kernel_function, class_weight=class_weights)

print("Training the weighted SVM with a quantum kernel...")
svc.fit(X_train, y_train)

# 4. Evaluate the weighted SVM
print("\nEvaluating the weighted SVM...")
svc_score = svc.score(X_test, y_test)
print(f"Weighted SVM score on the test set: {svc_score:.4f}")

# Compare with an unweighted SVM
print("\nFor comparison, training an unweighted SVM...")
svc_unweighted = SVC(kernel=quantum_kernel_function)
svc_unweighted.fit(X_train, y_train)
svc_unweighted_score = svc_unweighted.score(X_test, y_test)
print(f"Unweighted SVM score on the test set: {svc_unweighted_score:.4f}")
