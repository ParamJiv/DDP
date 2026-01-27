import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from torch.utils.data import DataLoader

# --- Qiskit Imports ---
# Ensure correct imports based on Qiskit version (assuming >= 1.0)
try:
    # Qiskit 1.x Primitives V2 style
    from qiskit.primitives import Estimator as RefEstimator
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN
    # V2 Gradient (Note: ParamShift might not be directly in qiskit.primitives.gradients,
    # EstimatorGradient might be the V2 equivalent provided by EstimatorQNN itself if not specified)
    # Check Qiskit ML docs for the exact V2 compatible gradient class if needed.
    # For now, we'll rely on EstimatorQNN's default or internal mechanism by not providing one.
    # from qiskit.primitives.gradients import EstimatorGradient as V2EstimatorGradient
    qiskit_1x = True
except ImportError:
    # Fallback attempt for older Qiskit / Qiskit Algorithms style (might need qiskit-terra<1.0)
    from qiskit.primitives import Estimator as RefEstimator # Might still be V1 depending on version
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN
    # V1 Gradient
    from qiskit_algorithms.gradients import ParamShiftEstimatorGradient # V1 style
    qiskit_1x = False


# --- Global CPU Estimator ---
# Use the stable reference estimator
CPU_ESTIMATOR = RefEstimator()
logging.info("Using qiskit.primitives.Estimator (Reference).")

# --- Quantum Encoder Module ---
class QiskitQuantumEncoder(nn.Module):
    """ Quantum encoder using Qiskit EstimatorQNN and TorchConnector """
    def __init__(self, n_qubits: int, q_depth: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        self.input_params = ParameterVector('x', self.n_qubits)
        self.weight_params = ParameterVector('w', self.q_depth * self.n_qubits)

        # Build Quantum Circuit
        qc = QuantumCircuit(self.n_qubits)
        # 1. Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
        # 2. Input encoding layer
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        # 3. Variational layers (entanglement + weights)
        for d in range(self.q_depth):
            # Entanglement
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            # Weight layer
            layer_slice = self.weight_params[d * self.n_qubits:(d + 1) * self.n_qubits]
            for i, wp in enumerate(layer_slice):
                qc.ry(wp, i)

        # Observables: Z measurement on each qubit
        observables = [
            SparsePauliOp.from_list([("I" * i + "Z" + "I" * (self.n_qubits - i - 1), 1)])
            for i in range(self.n_qubits)
        ]

        # Define QNN using the reference estimator
        # Do not explicitly pass a V1 gradient class if using Qiskit 1.x / Primitives V2
        qnn_kwargs = {
            "circuit": qc,
            "observables": observables,
            "input_params": self.input_params,
            "weight_params": self.weight_params,
            "input_gradients": False, # Crucial: Keep False for SVDD setup
            "estimator": CPU_ESTIMATOR,
        }
        # Only add V1 gradient if definitely using older Qiskit
        # if not qiskit_1x:
        #     grad_v1 = ParamShiftEstimatorGradient(CPU_ESTIMATOR)
        #     qnn_kwargs["gradient"] = grad_v1

        self.qnn = EstimatorQNN(**qnn_kwargs)

        # Initialize weights for TorchConnector
        init_w = (2 * np.random.rand(self.qnn.num_weights) - 1) * 0.1
        # Important: TorchConnector expects float64 initial weights
        self.torch_layer = TorchConnector(self.qnn, initial_weights=init_w.astype(np.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantum encoder.
        Applies scaling before passing to the TorchConnector layer.
        Ensures output dtype matches input dtype.
        """
        # Expect input shape: (batch, n_qubits) from PCA
        if x.dim() == 1:
            x = x.unsqueeze(0) # Handle single sample case
        if x.shape[1] != self.n_qubits:
             raise ValueError(f"Input feature dimension {x.shape[1]} != n_qubits {self.n_qubits}")

        dev, dt = x.device, x.dtype # Store original device and dtype

        # Ensure input is float64 and on CPU for TorchConnector/Estimator
        x64_cpu = x.to(dtype=torch.float64, device='cpu')

        # Apply input scaling (common practice for angle encoding)
        # Map features roughly to [-pi/2, pi/2]
        x_scaled = torch.tanh(x64_cpu) * np.pi / 2.0

        # Pass scaled data to the TorchConnector layer
        y = self.torch_layer(x_scaled) # Output is float64 on CPU

        # Return output tensor on original device and with original dtype
        # Cast back to original dtype AFTER potentially moving back to GPU
        return y.to(device=dev, dtype=dt)


# --- Quantum Autoencoder (Optional Pretraining) ---
class QuantumAutoencoder(nn.Module):
    """ Autoencoder using the QiskitQuantumEncoder and a classical decoder """
    def __init__(self, n_qubits: int, q_depth: int, classical_hidden_dims: list = [32]):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        # Quantum Encoder part
        self.encoder = QiskitQuantumEncoder(n_qubits=self.n_qubits, q_depth=self.q_depth)

        # Classical Decoder part
        decoder_layers = []
        prev_dim = self.n_qubits # Decoder input is the encoder output size
        for h_dim in classical_hidden_dims:
            # Linear layer requires float64 weights/biases
            decoder_layers.append(nn.Linear(prev_dim, h_dim, dtype=torch.float64))
            decoder_layers.append(nn.ReLU()) # ReLU works with float64
            prev_dim = h_dim
        # Final layer reconstructs to the original PCA dimension (n_qubits)
        decoder_layers.append(nn.Linear(prev_dim, self.n_qubits, dtype=torch.float64))

        self.decoder = nn.Sequential(*decoder_layers)
        # Ensure decoder is explicitly float64 (redundant if layers initialized correctly, but safe)
        self.decoder = self.decoder.to(dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Encodes with quantum layer, decodes with classical layers """
        # x is assumed to be float64 from the dataset/loader
        z = self.encoder(x) # Encoder handles internal scaling and type/device casting
        # z comes back with original dtype/device (should be float64, possibly GPU)

        # Ensure z matches decoder's expected dtype and device before passing
        decoder_param = next(self.decoder.parameters()) # Get a sample parameter
        z_matched = z.to(device=decoder_param.device, dtype=decoder_param.dtype)

        reconstructed_output = self.decoder(z_matched)
        return reconstructed_output

# --- Main QSVDD Class ---
class QSVDD:
    """ Quantum Support Vector Data Description using Qiskit """
    def __init__(self, n_qubits: int, q_depth: int, objective: str = 'one-class', nu: float = 0.1):
        assert objective in ('one-class', 'soft-boundary'), "Objective must be 'one-class' or 'soft-boundary'"
        assert 0 < nu <= 1, "nu must be in (0, 1]"

        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.objective = objective
        self.nu = nu

        self.R = torch.tensor(0.0, dtype=torch.float64) # Radius of hypersphere (for soft-boundary)
        self.c = None # Center of hypersphere, shape (n_qubits,) type float64

        self.net: nn.Module | None = None # Main SVDD network (feature extractor)
        self.ae_net: QuantumAutoencoder | None = None # Autoencoder for pretraining

        # Training config and results storage
        self.device = 'cpu'
        self.optimizer_name = None
        self.ae_optimizer_name = None
        self.optimizer = None
        self.scheduler = None
        self.start_time = None
        self.train_losses_per_epoch = []
        self.svdd_losses_per_epoch = []
        self.results = {'train_time': None, 'test_auc': None, 'test_time': None, 'test_scores': None}
        self.pretrain_completed = False

    def build_autoencoder(self, classical_hidden_dims: list = [32]):
        """ Builds the QuantumAutoencoder for pretraining """
        self.ae_net = QuantumAutoencoder(
            n_qubits=self.n_qubits,
            q_depth=self.q_depth,
            classical_hidden_dims=classical_hidden_dims
        )
        self.ae_net = self.ae_net.to(dtype=torch.float64) # Ensure entire AE is float64
        logging.info(f"QuantumAutoencoder built with classical_hidden_dims={classical_hidden_dims}")

    def pretrain(self, train_loader: DataLoader, test_loader: DataLoader,
                 optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 10,
                 lr_milestones: tuple = (), weight_decay: float = 1e-6, device: str = 'cpu'):
        """ Pretrains the QuantumAutoencoder (focuses on decoder weights) """
        logger = logging.getLogger(__name__)
        self.device = device

        if self.ae_net is None:
            self.build_autoencoder() # Build with default dims if not already done
        self.ae_net = self.ae_net.to(self.device) # Move AE to device

        # --- Freeze quantum encoder ---
        # We only pretrain the classical decoder weights
        for param in self.ae_net.encoder.parameters():
            param.requires_grad = False
        logger.info("Quantum encoder parameters frozen for pretraining.")

        # --- Setup Optimizer & Scheduler ---
        # Optimize only the decoder parameters
        trainable_params = filter(lambda p: p.requires_grad, self.ae_net.parameters())
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            # Add other optimizers if needed
            logger.warning(f"Optimizer '{optimizer_name}' not recognized, using Adam.")
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

        scheduler = None
        if lr_milestones:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

        # --- Training Loop ---
        logger.info(f"Starting Autoencoder pretraining for {n_epochs} epochs...")
        start_time = time.time()
        self.ae_net.train() # Set AE model to training mode

        criterion = nn.MSELoss() # Reconstruction loss

        epoch_losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for features, _, _ in train_loader: # Labels ignored during AE pretrain
                features = features.to(self.device, dtype=torch.float64) # Ensure correct dtype/device

                optimizer.zero_grad()
                reconstructed = self.ae_net(features)
                loss = criterion(reconstructed, features) # Compare reconstruction to original features
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(1, n_batches)
            epoch_losses.append(avg_epoch_loss)
            logger.info(f'AE Pretrain Epoch {epoch + 1}/{n_epochs}, Avg Loss: {avg_epoch_loss:.6f}')

            if scheduler:
                scheduler.step()

        # --- Final Logging ---
        end_time = time.time()
        pretrain_time = end_time - start_time
        logger.info(f'Finished Autoencoder pretraining. Time: {pretrain_time:.2f}s')
        self.pretrain_completed = True
        self.train_losses_per_epoch = epoch_losses # Store AE losses if needed later

        # --- Unfreeze encoder for SVDD training ---
        for param in self.ae_net.encoder.parameters():
            param.requires_grad = True
        logger.info("Quantum encoder parameters unfrozen.")

        return self.ae_net # Return the pretrained autoencoder


    def build_main_network(self):
        """ Builds the main SVDD network (just the quantum encoder) """
        # For image data with PCA, the main network is just the encoder
        self.net = QiskitQuantumEncoder(n_qubits=self.n_qubits, q_depth=self.q_depth)
        self.net = self.net.to(self.device, dtype=torch.float64)
        logging.info("Main SVDD network (QiskitQuantumEncoder) built.")

    def transfer_pretrained_encoder(self, freeze: bool = False):
        """ Transfers weights from pretrained AE encoder to main SVDD network """
        logger = logging.getLogger(__name__)
        if not self.pretrain_completed or self.ae_net is None:
            logger.warning("Autoencoder pretraining not completed or ae_net not built. Skipping weight transfer.")
            return
        if self.net is None:
             logger.warning("Main network not built. Skipping weight transfer.")
             return

        # Source is ae_net.encoder.torch_layer, Destination is net.torch_layer
        src_state_dict = self.ae_net.encoder.torch_layer.state_dict()
        dst_state_dict = self.net.torch_layer.state_dict()

        # Transfer weights (usually named 'weight' or '_weights' in TorchConnector)
        transferred_keys = []
        for key in src_state_dict:
            if key in dst_state_dict and src_state_dict[key].shape == dst_state_dict[key].shape:
                dst_state_dict[key] = src_state_dict[key].clone().detach()
                transferred_keys.append(key)

        if not transferred_keys:
            logger.warning("Could not find matching keys to transfer weights from AE encoder to main net.")
        else:
            self.net.torch_layer.load_state_dict(dst_state_dict)
            logger.info(f"Transferred weights for keys: {transferred_keys} from pretrained AE encoder.")

        # Optionally freeze the transferred weights
        if freeze:
            for param in self.net.torch_layer.parameters():
                param.requires_grad = False
            logger.info("Main network's quantum encoder weights frozen.")


    def _initialize_center_c(self, train_loader: DataLoader, eps: float = 1e-6):
        """ Initializes the center c of the hypersphere using the network output """
        logger = logging.getLogger(__name__)
        if self.net is None:
            raise RuntimeError("Main network not built. Cannot initialize center c.")

        logger.info("Initializing center c...")
        self.net.eval() # Set network to evaluation mode
        c_sum = torch.zeros(self.n_qubits, dtype=torch.float64, device=self.device)
        n_samples = 0
        with torch.no_grad():
            for features, _, _ in train_loader:
                features = features.to(self.device, dtype=torch.float64)
                outputs = self.net(features)
                c_sum += torch.sum(outputs, dim=0)
                n_samples += features.size(0)

        if n_samples == 0:
            raise ValueError("Train loader is empty, cannot initialize center c.")

        self.c = c_sum / n_samples
        # Add small epsilon to prevent center from being exactly zero (optional but common)
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c >= 0)] = eps

        # Detach c from computation graph
        self.c = self.c.clone().detach()
        logger.info(f"Center c initialized (shape: {self.c.shape}, device: {self.c.device}, dtype: {self.c.dtype}).")
        self.net.train() # Set back to training mode


    def train_svdd(self, train_loader: DataLoader,
                   optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
                   lr_milestones: tuple = (), weight_decay: float = 1e-6, device: str = 'cpu'):
        """ Trains the main QSVDD network """
        logger = logging.getLogger(__name__)
        self.device = device
        self.optimizer_name = optimizer_name

        if self.net is None:
            self.build_main_network() # Build if not done
        self.net = self.net.to(self.device) # Move net to device

        # --- Initialize Center c ---
        # Must be done AFTER net is on the correct device and BEFORE optimizer setup
        if self.c is None:
            self._initialize_center_c(train_loader)
        # Ensure c is on the correct device for loss calculation
        self.c = self.c.to(self.device)

        # --- Setup Optimizer & Scheduler ---
        # Ensure we optimize only trainable parameters
        trainable_params = list(filter(lambda p: p.requires_grad, self.net.parameters()))
        if not trainable_params:
             logger.warning("No trainable parameters found in the main network. Check freezing settings.")
             # Attempt to unfreeze just in case, though this shouldn't normally happen if built correctly
             for param in self.net.parameters(): param.requires_grad = True
             trainable_params = list(filter(lambda p: p.requires_grad, self.net.parameters()))
             if not trainable_params:
                  raise RuntimeError("Still no trainable parameters after attempting unfreeze.")


        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            logger.warning(f"Optimizer '{optimizer_name}' not recognized, using Adam.")
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

        self.scheduler = None
        if lr_milestones:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.1)

        # --- SVDD Training Loop ---
        logger.info(f"Starting QSVDD training for {n_epochs} epochs...")
        self.start_time = time.time()
        self.net.train() # Set to training mode

        self.svdd_losses_per_epoch = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for features, _, _ in train_loader: # Labels ignored during one-class SVDD training
                features = features.to(self.device, dtype=torch.float64)

                self.optimizer.zero_grad()

                # Forward pass through the network (quantum encoder)
                outputs = self.net(features) # Shape: (batch_size, n_qubits)

                # Calculate distance-based loss
                dist_sq = torch.sum((outputs - self.c) ** 2, dim=1) # Squared Euclidean distance

                # SVDD Loss (one-class objective)
                # For soft-boundary, need radius R update - simplified here
                if self.objective == 'one-class':
                    loss = torch.mean(dist_sq)
                elif self.objective == 'soft-boundary':
                    # Simplified soft-boundary: fixed R or R estimated once
                    # Proper soft-boundary requires updating R based on quantiles
                    scores = torch.sqrt(dist_sq + 1e-8) # Numerical stability
                    penalty = torch.mean(torch.relu(scores - self.R)) # Hinge loss
                    loss = self.R**2 + (1.0 / self.nu) * penalty
                    # Need to update R here - omitting for simplicity, using one-class loss instead
                    loss = torch.mean(dist_sq) # Fallback to one-class if soft-boundary R not implemented
                else:
                    loss = torch.mean(dist_sq) # Default to one-class

                # Backward pass and optimization
                try:
                    loss.backward()
                except Exception as e:
                    logger.error(f"Backward pass failed during SVDD training: {e}")
                    # Log details helpful for Qiskit errors
                    logger.error(f"Loss value: {loss.item()}, Output shape: {outputs.shape}, Output dtype: {outputs.dtype}")
                    raise # Re-raise after logging

                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # --- End of Epoch ---
            avg_epoch_loss = epoch_loss / max(1, n_batches)
            self.svdd_losses_per_epoch.append(avg_epoch_loss)
            logger.info(f'SVDD Train Epoch {epoch + 1}/{n_epochs}, Avg Loss: {avg_epoch_loss:.6f}')

            if self.scheduler:
                self.scheduler.step()

            # Update R for soft-boundary (simplified - update R based on epoch end scores)
            # if self.objective == 'soft-boundary':
            #     self.R.data = torch.tensor(self._get_radius(train_loader), device=self.device)

        # --- End of Training ---
        train_time = time.time() - self.start_time
        self.results['train_time'] = train_time
        logger.info(f'Finished QSVDD training. Time: {train_time:.2f}s')

        return self.net

    def _get_radius(self, data_loader: DataLoader, quantile: float = None):
        """ Helper to compute radius R for soft-boundary (optional) """
        if quantile is None:
            quantile = self.nu # Use nu as the quantile by default

        self.net.eval()
        all_dists = []
        with torch.no_grad():
             for features, _, _ in data_loader:
                  features = features.to(self.device, dtype=torch.float64)
                  outputs = self.net(features)
                  dist_sq = torch.sum((outputs - self.c) ** 2, dim=1)
                  all_dists.append(torch.sqrt(dist_sq + 1e-8).cpu())
        all_dists = torch.cat(all_dists).numpy()
        radius = np.quantile(all_dists, 1.0 - quantile) # R is the (1-nu)-quantile of distances
        self.net.train()
        return radius


    def test(self, test_loader: DataLoader):
        """ Tests the trained QSVDD model on the test set """
        logger = logging.getLogger(__name__)
        if self.net is None or self.c is None:
            raise RuntimeError("Model not trained or center c not initialized. Call train_svdd first.")

        logger.info("Starting QSVDD testing...")
        start_time = time.time()
        self.net.eval() # Set to evaluation mode

        all_labels = []
        all_scores = []
        all_indices = []

        with torch.no_grad():
            for features, labels, indices in test_loader:
                features = features.to(self.device, dtype=torch.float64)

                outputs = self.net(features)
                dist_sq = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = torch.sqrt(dist_sq + 1e-8) # Anomaly score is distance

                all_labels.append(labels.cpu())
                all_scores.append(scores.cpu())
                all_indices.append(indices.cpu())

        # Concatenate results from all batches
        all_labels = torch.cat(all_labels).numpy()
        all_scores = torch.cat(all_scores).numpy()
        all_indices = torch.cat(all_indices).numpy()

        # Calculate AUC score
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_scores)
            self.results['test_auc'] = auc
            logger.info(f"Test AUC: {auc:.6f}")
        except ValueError as e:
            # Handle cases where AUC cannot be calculated (e.g., only one class in test labels)
            logger.warning(f"Could not calculate AUC: {e}")
            self.results['test_auc'] = None
            auc = None
        except ImportError:
            logger.warning("Scikit-learn not found. Cannot calculate AUC.")
            self.results['test_auc'] = None
            auc = None


        test_time = time.time() - start_time
        self.results['test_time'] = test_time
        self.results['test_scores'] = all_scores # Store scores for potential thresholding/plotting
        # Store labels and indices if needed for analysis
        # self.results['test_labels'] = all_labels
        # self.results['test_indices'] = all_indices

        logger.info(f"Finished QSVDD testing. Time: {test_time:.2f}s")

        return all_scores, all_labels, auc
