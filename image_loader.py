import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("./data") # Directory to store downloaded datasets
DATA_DIR.mkdir(exist_ok=True)

# --- Custom Dataset Class ---
class ImageAnomalyDataset(Dataset):
    """
    Custom Dataset for anomaly detection on image datasets after PCA.
    Expects features that are already PCA-transformed.
    Labels are binary: 0 for normal, 1 for anomaly.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
             raise TypeError("Features and labels must be numpy arrays.")
        # Ensure float64 for Qiskit compatibility
        self.features = torch.tensor(features, dtype=torch.float64)
        # Ensure long for potential loss functions later, keep labels binary (0/1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = torch.arange(len(features)) # Add indices

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return features, label (0 or 1), and original index
        return self.features[idx], self.labels[idx], self.indices[idx]

# --- Common Preprocessing Function ---
def _preprocess_image_data(
    dataset_name: str,
    train_data,
    test_data,
    normal_class_label: int,
    n_components: int,
    batch_size: int,
    num_workers: int = 0
):
    """
    Internal helper to preprocess image data for anomaly detection using QSVDD.

    Args:
        dataset_name: Name for logging ('MNIST' or 'CIFAR').
        train_data: Raw torchvision training dataset.
        test_data: Raw torchvision test dataset.
        normal_class_label: The integer label of the class considered normal.
        n_components: Number of PCA components (should match n_qubits).
        batch_size: DataLoader batch size.
        num_workers: DataLoader num_workers.

    Returns:
        tuple: (train_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing {dataset_name} data...")

    # --- 1. Filter Data ---
    # Training data: only normal class
    train_indices = [i for i, (_, label) in enumerate(train_data) if label == normal_class_label]
    X_train_raw = np.array([train_data[i][0].numpy() for i in train_indices])
    # Labels for training are all 0 (normal), though not strictly needed for SVDD training itself
    y_train = np.zeros(len(X_train_raw), dtype=int)

    # Test data: all classes, binary labels (0=normal, 1=anomaly)
    X_test_raw = np.array([data[0].numpy() for data in test_data])
    y_test = np.array([0 if label == normal_class_label else 1 for _, label in test_data], dtype=int)

    logger.info(f"Data filtered. Normal class: {normal_class_label}.")
    logger.info(f"Raw train shape: {X_train_raw.shape}, Raw test shape: {X_test_raw.shape}")
    logger.info(f"Anomaly counts - Train: {np.sum(y_train)}, Test: {np.sum(y_test)}/{len(y_test)}")

    # --- 2. Flatten Images ---
    original_shape = X_train_raw.shape[1:]
    flattened_dim = np.prod(original_shape)
    X_train_flat = X_train_raw.reshape(len(X_train_raw), -1)
    X_test_flat = X_test_raw.reshape(len(X_test_raw), -1)
    logger.info(f"Images flattened from {original_shape} to ({flattened_dim},)")

    # --- 3. Scale Data ---
    # Fit scaler ONLY on normal training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat) # Apply same scaling to test data
    logger.info("Data scaled using StandardScaler (fitted on train).")

    # --- 4. Apply PCA ---
    # Fit PCA ONLY on scaled normal training data
    if n_components > X_train_scaled.shape[1]:
        logger.warning(f"n_components ({n_components}) > flattened dim ({X_train_scaled.shape[1]}). Setting n_components to {X_train_scaled.shape[1]}.")
        n_components = X_train_scaled.shape[1]
    elif n_components > X_train_scaled.shape[0]:
         logger.warning(f"n_components ({n_components}) > n_train_samples ({X_train_scaled.shape[0]}). Setting n_components to {X_train_scaled.shape[0]}.")
         n_components = X_train_scaled.shape[0]


    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled) # Apply same PCA to test data
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA applied. Reduced dim to {n_components}. Explained variance: {explained_variance:.4f}")

    # --- 5. Create Datasets and DataLoaders ---
    train_dataset = ImageAnomalyDataset(features=X_train_pca, labels=y_train)
    test_dataset = ImageAnomalyDataset(features=X_test_pca, labels=y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) # Don't shuffle test

    logger.info("Datasets and DataLoaders created.")

    return train_loader, test_loader, original_shape

# --- MNIST Loader Function ---
def get_mnist_dataloaders(
    batch_size: int,
    normal_class: int = 0,
    n_components: int = 10, # Number of PCA components (== n_qubits)
    num_workers: int = 0
):
    """
    Creates DataLoaders for MNIST anomaly detection.

    Args:
        batch_size: Size of batches for DataLoader.
        normal_class: The digit (0-9) to be treated as the normal class.
        n_components: Number of principal components for dimensionality reduction.
        num_workers: Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, test_loader, original_image_shape)
    """
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0, 1] range
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific mean/std
    ])

    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    return _preprocess_image_data(
        "MNIST", train_set, test_set, normal_class, n_components, batch_size, num_workers
    )

# --- CIFAR-10 Loader Function ---
def get_cifar_dataloaders(
    batch_size: int,
    normal_class: int = 0, # 0: airplane, 1: automobile, ..., 9: truck
    n_components: int = 10, # Number of PCA components (== n_qubits)
    num_workers: int = 0
):
    """
    Creates DataLoaders for CIFAR-10 anomaly detection.

    Args:
        batch_size: Size of batches for DataLoader.
        normal_class: The class index (0-9) to be treated as the normal class.
        n_components: Number of principal components for dimensionality reduction.
        num_workers: Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, test_loader, original_image_shape)
    """
    # CIFAR-10 specific mean/std for normalization
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts image to [C, H, W] tensor in [0, 1]
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    return _preprocess_image_data(
        "CIFAR", train_set, test_set, normal_class, n_components, batch_size, num_workers
    )

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    N_QUBITS = 10 # Must match n_components
    BATCH_SIZE = 128

    print("\n--- Testing MNIST Loader ---")
    mnist_train_loader, mnist_test_loader, mnist_shape = get_mnist_dataloaders(
        batch_size=BATCH_SIZE, normal_class=0, n_components=N_QUBITS
    )
    print(f"MNIST original shape: {mnist_shape}")
    # Inspect a batch
    train_features, train_labels, train_indices = next(iter(mnist_train_loader))
    print(f"MNIST Train Batch - Features shape: {train_features.shape}, dtype: {train_features.dtype}")
    print(f"MNIST Train Batch - Labels sample: {train_labels[:5]}, dtype: {train_labels.dtype}")
    print(f"MNIST Train Batch - Indices sample: {train_indices[:5]}")

    test_features, test_labels, test_indices = next(iter(mnist_test_loader))
    print(f"MNIST Test Batch - Features shape: {test_features.shape}, dtype: {test_features.dtype}")
    print(f"MNIST Test Batch - Labels sample: {test_labels[:10]}, dtype: {test_labels.dtype}") # Show more to see anomalies
    print(f"MNIST Test Batch - Indices sample: {test_indices[:5]}")

    print("\n--- Testing CIFAR-10 Loader ---")
    cifar_train_loader, cifar_test_loader, cifar_shape = get_cifar_dataloaders(
        batch_size=BATCH_SIZE, normal_class=1, n_components=N_QUBITS # Use class 1 (automobile) as normal
    )
    print(f"CIFAR-10 original shape: {cifar_shape}")
    # Inspect a batch
    train_features_c, train_labels_c, _ = next(iter(cifar_train_loader))
    print(f"CIFAR Train Batch - Features shape: {train_features_c.shape}, dtype: {train_features_c.dtype}")
    print(f"CIFAR Train Batch - Labels sample: {train_labels_c[:5]}")

    test_features_c, test_labels_c, _ = next(iter(cifar_test_loader))
    print(f"CIFAR Test Batch - Features shape: {test_features_c.shape}, dtype: {test_features_c.dtype}")
    print(f"CIFAR Test Batch - Labels sample: {test_labels_c[:10]}")
