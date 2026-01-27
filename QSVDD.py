import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import logging

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve
)

import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split

from abc import ABC, abstractmethod
import time

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# from qiskit.utils import QuantumInstance
# from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.connectors import TorchConnector


from torch.utils.data import DataLoader, Dataset, random_split
class FraudDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = torch.arange(len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.indices[idx]


data = pd.read_csv(r"D:\Academic\QML Intern\Anomaly Detection\Fraud Detection\creditcard_data.csv")


def get_fraud_dataloaders(data_path: str, batch_size: int, n_fraud_points: int = 200, n_non_fraud_points: int = 600, n_components: int = 10):
    data = pd.read_csv(data_path)
    
    data_fraud = data.loc[data['Class'] == 1]
    data_non_fraud = data.loc[data['Class'] == 0]

    data_fraud_train = data_fraud.sample(n=n_fraud_points, random_state=42)
    data_fraud_test = data_fraud.drop(index=data_fraud_train.index)
    data_non_fraud_train = data_non_fraud.sample(n=n_non_fraud_points, random_state=42)
    data_non_fraud_test = data_non_fraud.drop(index=data_non_fraud_train.index)

    data_train = pd.concat([data_non_fraud_train, data_fraud_train])
    data_test = pd.concat([data_non_fraud_test, data_fraud_test])

    x_train = data_train.drop(['Time', 'Class'], axis=1).values
    y_train = data_train['Class'].values
    x_test = data_test.drop(['Time', 'Class'], axis=1).values
    y_test = data_test['Class'].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    
    print(f"Data processed. Train shape: {x_train_pca.shape}, Test shape: {x_test_pca.shape}")

    train_dataset = FraudDataset(features=x_train_pca, labels=y_train)
    test_dataset = FraudDataset(features=x_test_pca, labels=y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


n_qubits = 10
q_depth = 1
q_delta = 0.01
n_layers = 1
dim_lis = [32]


def H_layer_qiskit(qc: QuantumCircuit, n_qubits: int):
    for idx in range(n_qubits):
        qc.h(idx)

def RY_layer_qiskit(qc: QuantumCircuit, params: ParameterVector):
    for i, param in enumerate(params):
        qc.ry(param, i)

def entangling_layer_qiskit(qc: QuantumCircuit, n_qubits: int):
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n_qubits - 1, 2):
        qc.cx(i, i + 1)

def create_qiskit_qnn(n_qubits: int, q_depth: int):
    feature_params = ParameterVector('x', n_qubits)
    weight_params = ParameterVector('w', q_depth * n_qubits)

    qc = QuantumCircuit(n_qubits)

    H_layer_qiskit(qc, n_qubits)

    RY_layer_qiskit(qc, feature_params)

    for k in range(q_depth):
        entangling_layer_qiskit(qc, n_qubits)
        layer_weights = weight_params[k * n_qubits : (k + 1) * n_qubits]
        RY_layer_qiskit(qc, layer_weights)

    observables = [SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n_qubits - 1 - i), 1)]) for i in range(n_qubits)]

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=feature_params,
        weight_params=weight_params
    )

    initial_weights = (2 * np.random.rand(qnn.num_weights) - 1) * 0.1
    torch_qnn = TorchConnector(qnn, initial_weights=initial_weights)
    return torch_qnn


class QuantumAutoencoder(nn.Module):
    def __init__(self, n_qubits: int, q_depth: int, n_layers : int, dim_lis : list):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        self.encoder = create_qiskit_qnn(n_qubits=self.n_qubits, q_depth=self.q_depth)

        decoder_layers = []
        if n_layers == 0:
            decoder_layers.append(nn.Linear(self.n_qubits, self.n_qubits))
        else:
            decoder_layers.append(nn.Linear(self.n_qubits, dim_lis[0]))
            decoder_layers.append(nn.ReLU())

            for i in range(1, n_layers):
                decoder_layers.append(nn.Linear(dim_lis[i - 1], dim_lis[i]))
                decoder_layers.append(nn.ReLU())

            decoder_layers.append(nn.Linear(dim_lis[-1], self.n_qubits))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_representation = self.encoder(x)
        reconstructed_output = self.decoder(latent_representation)
        return reconstructed_output
    

# --- Qiskit quantum encoder (replacement for PennyLane Quantumnet) ---

class QiskitQuantumEncoder(nn.Module):
    def __init__(self, n_qubits: int, q_depth: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        # Build parameter vectors
        self.input_params = ParameterVector('x', self.n_qubits)
        self.weight_params = ParameterVector('w', self.q_depth * self.n_qubits)

        # Build variational circuit
        qc = QuantumCircuit(self.n_qubits)

        # Layer 0: Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        # Feature map: one RY per qubit with input parameter
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)

        # Variational layers
        for d in range(self.q_depth):
            # Entangling even pairs
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            # Entangling odd pairs
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            # Trainable single-qubit rotations
            layer_slice = self.weight_params[d * self.n_qubits:(d + 1) * self.n_qubits]
            for i, wp in enumerate(layer_slice):
                qc.ry(wp, i)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp.from_list([("I" * i + "Z" + "I" * (self.n_qubits - i - 1), 1)])
            for i in range(self.n_qubits)
        ]

        # EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=self.input_params,
            weight_params=self.weight_params
        )

        # Torch connector with small random init
        init_w = (2 * np.random.rand(self.qnn.num_weights) - 1) * 0.1
        self.torch_layer = TorchConnector(self.qnn, initial_weights=init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape: (batch, n_qubits)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == self.n_qubits, f"Input feature dimension {x.size(1)} != n_qubits {self.n_qubits}"
        # Optional scaling similar to your tanh mapping
        x_scaled = torch.tanh(x) * np.pi / 2.0
        return self.torch_layer(x_scaled)

class ParametricHybridNet(nn.Module):
    def __init__(self,input_shape=(1, 28, 28),classical_dims=None,n_qubits: int = 10, q_depth: int = 1,activation: str = 'relu'):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.rep_dim = n_qubits
        classical_dims = classical_dims or []  # list[int]

        act_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }

        act_layer = act_map.get(activation.lower(), nn.ReLU())

        in_features = 1
        for v in input_shape:
            in_features *= v

        layers = []
        prev = in_features
        for h in classical_dims:
            layers.append(nn.Linear(prev, h, bias=False))
            layers.append(act_layer)
            prev = h

        if prev != n_qubits:
            layers.append(nn.Linear(prev, n_qubits, bias=False))

        self.classical_feature_map = nn.Sequential(*layers)
        self.qnn = QiskitQuantumEncoder(n_qubits=n_qubits, q_depth=q_depth)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = self.classical_feature_map(x)
        x = self.qnn(x)  # shape (B, n_qubits)
        return x

# Example instantiation (adjust as needed):
# hybrid_net = ParametricHybridNet(classical_dims=[256,128,64], n_qubits=10, q_depth=3)

def get_network(classical_dims=None,n_qubits: int = n_qubits,q_depth: int = q_depth,activation: str = 'relu',input_shape=(1, 28, 28)):
    return ParametricHybridNet(input_shape=input_shape,classical_dims=classical_dims or [128, 64],n_qubits=n_qubits,q_depth=q_depth,activation=activation)

def build_network(classical_dims=None,n_qubits: int = n_qubits,q_depth: int = q_depth,activation: str = 'relu',input_shape=(1, 28, 28)):
    return get_network(classical_dims, n_qubits, q_depth, activation, input_shape)

def get_ae_network(n_qubits=n_qubits, q_depth=q_depth, n_layers=n_layers, dim_lis=dim_lis):
    ae_net = QuantumAutoencoder(n_qubits=n_qubits, q_depth=q_depth, n_layers=n_layers, dim_lis=dim_lis)
    return ae_net

def build_autoencoder(n_qubits=n_qubits, q_depth=q_depth, n_layers=n_layers, dim_lis=dim_lis):
    ae_net = get_ae_network(n_qubits=n_qubits, q_depth=q_depth, n_layers=n_layers, dim_lis=dim_lis)
    return ae_net

DATA_PATH = r"D:\Academic\QML Intern\Anomaly Detection\Fraud Detection\creditcard_data.csv"

train_loader, test_loader = get_fraud_dataloaders(data_path=DATA_PATH, batch_size=128)

# ...existing code above...

class QSVDD:
    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        assert objective in ('one-class', 'soft-boundary')
        assert 0 < nu <= 1
        self.objective = objective
        self.nu = nu
        self.R = 0.0
        self.c = None

        self.net_name = None
        self.net = None

        # Autoencoder / pretraining related attributes
        self.ae_net = None
        self.device = 'cpu'
        self.n_epochs = 0
        self.lr_milestones = ()
        self.optimizer_name = None
        self.ae_optimizer_name = None

        # Will be populated during pretrain()
        self.optimizer = None
        self.scheduler = None
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.current_epoch = 0
        self.batch_size = None
        self.weight_decay = None
        self.train_losses_per_epoch = []
        self.train_losses_per_batch = []
        self.avg_train_loss = None
        self.avg_test_loss = None
        self.last_epoch_train_loss = None
        self.last_epoch_batches = 0
        self.test_loss_epoch = 0.0
        self.test_batches = 0

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def pretrain(self,train_loader: DataLoader,test_loader: DataLoader,optimizer_name: str = 'adam',lr: float = 0.001,n_epochs: int = 150,lr_milestones: tuple = (),batch_size: int = 128,weight_decay: float = 1e-6,device: str = 'cuda',n_jobs_dataloader: int = 0):

        logger = logging.getLogger(__name__)

        # Store config as attributes
        self.device = device
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.ae_optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.train_losses_per_epoch = []
        self.train_losses_per_batch = []
        self.avg_train_loss = None
        self.avg_test_loss = None

        # Build / move model
        if self.ae_net is None:
            self.ae_net = get_ae_network()
        self.ae_net.to(self.device)

        # Optimizer & scheduler as attributes
        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=lr_milestones,
                                                        gamma=0.1) if lr_milestones else None

        logger.info("Starting pretraining...")
        self.start_time = time.time()

        # Training loop
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch + 1
            self.ae_net.train()
            epoch_loss_sum = 0.0
            n_batches = 0

            for batch in train_loader:
                inputs, _, _ = batch
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                self.optimizer.step()

                epoch_loss_sum += loss.item()
                n_batches += 1
                self.train_losses_per_batch.append(loss.item())

            epoch_avg = epoch_loss_sum / max(1, n_batches)
            self.train_losses_per_epoch.append(epoch_avg)
            self.last_epoch_train_loss = epoch_avg
            self.last_epoch_batches = n_batches

            logger.info(f'Epoch {epoch + 1}/{self.n_epochs}  Train Loss: {epoch_avg:.8f}')

            if self.scheduler:
                self.scheduler.step()

        # Aggregate final training stats
        self.avg_train_loss = sum(self.train_losses_per_epoch) / len(self.train_losses_per_epoch)

        # Test phase
        self.ae_net.eval()
        self.test_loss_epoch = 0.0
        self.test_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, _, _ = batch
                inputs = inputs.to(self.device)
                outputs = self.ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                self.test_loss_epoch += loss.item()
                self.test_batches += 1

        self.avg_test_loss = self.test_loss_epoch / max(1, self.test_batches)

        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        logger.info(f'Finished pretraining. '
                    f'Avg Train Loss: {self.avg_train_loss:.8f} | '
                    f'Avg Test Loss: {self.avg_test_loss:.8f} | '
                    f'Time: {self.elapsed_time:.2f}s')

        return self.ae_net
    
    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()
        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """保存深度SVDD模型以导出_模型Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)
        print('R', self.R)
        print('c', self.c)

    def load_model(self, model_path, load_ae=False):
        """从模型路径加载深度SVDD模型Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def build_main_network(self,
                           classical_dims=None,
                           n_qubits: int = n_qubits,
                           q_depth: int = q_depth,
                           activation: str = 'relu',
                           input_shape=(1, 28, 28)):
        """
        Build main feature network (φ) after pretraining.
        """
        self.net = get_network(input_shape=input_shape,
            classical_dims=classical_dims or [128, 64],
            n_qubits=n_qubits,
            q_depth=q_depth,
            activation=activation).to(self.device)

    def transfer_pretrained_encoder(self, freeze: bool = False):
        """
        Robustly transfer TorchConnector (quantum encoder) weights from pretrained AE (self.ae_net.encoder)
        to main network (self.net.qnn.torch_layer).

        Handles different Qiskit versions:
          possible source/dest keys: 'weights', 'weight', '_weights'.
        Copies every matching-shape key that appears in both, else falls back to first shape match.
        """
        assert self.ae_net is not None, "Pretrain first."
        assert self.net is not None, "Build main network first."
        assert hasattr(self.net, 'qnn') and hasattr(self.net.qnn, 'torch_layer'), "Main net missing qnn.torch_layer."

        src = self.ae_net.encoder
        dst = self.net.qnn.torch_layer

        src_sd = src.state_dict()
        dst_sd = dst.state_dict()

        candidate_names = ['weights', 'weight', '_weights']

        # Collect matching keys
        matched = []
        for name in candidate_names:
            if name in src_sd and name in dst_sd and src_sd[name].shape == dst_sd[name].shape:
                dst_sd[name] = src_sd[name].clone()
                matched.append(name)

        # Fallback: if nothing matched, try any shape-equal pair
        if not matched:
            for sk, sv in src_sd.items():
                for dk, dv in dst_sd.items():
                    if sv.shape == dv.shape:
                        dst_sd[dk] = sv.clone()
                        matched.append(f"{sk}->{dk}")
                        break
                if matched:
                    break

        if not matched:
            raise RuntimeError(
                f"No transferable weight keys found. Source keys: {list(src_sd.keys())} "
                f"Dest keys: {list(dst_sd.keys())}"
            )

        dst.load_state_dict(dst_sd)

        if freeze:
            for p in dst.parameters():
                p.requires_grad = False

        logging.getLogger(__name__).info(
            f"Transferred pretrained quantum encoder weights. Mapped keys: {matched}. freeze={freeze}"
        )

    def train_svdd(self,
                   train_loader,
                   lr: float = 1e-3,
                   n_epochs: int = 20,
                   weight_decay: float = 1e-6,
                   freeze_quantum_epochs: int = 0):
        assert self.net is not None, "Build main network first."
        if self.c is None:
            self._initialize_center_c(train_loader)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        logger = logging.getLogger(__name__)
        logger.info("Starting Deep SVDD training...")

        quantum_initially_frozen = hasattr(self.net, 'qnn') and \
            any(not p.requires_grad for p in self.net.qnn.torch_layer.parameters())

        for epoch in range(1, n_epochs + 1):
            # Unfreeze after warmup
            if quantum_initially_frozen and freeze_quantum_epochs > 0 and epoch == freeze_quantum_epochs + 1:
                for p in self.net.qnn.torch_layer.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                optimizer.add_param_group({"params": self.net.qnn.torch_layer.parameters()})
                logger.info(f"Unfroze quantum weights at epoch {epoch}.")

            self.net.train()
            total_loss = 0.0
            batches = 0
            for x, _, _ in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                phi = self.net(x)
                dist = torch.sum((phi - self.c.to(self.device)) ** 2, dim=1)
                loss = dist.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

            logger.info(f"Epoch {epoch}/{n_epochs} SVDD Loss: {total_loss / batches:.6f}")

        logger.info("Deep SVDD training complete.")
        return self.net

    def _initialize_center_c(self, train_loader, eps=1e-6):
        """
        Initialize Deep SVDD center c as mean of embeddings.
        """
        self.net.eval()
        n = 0
        c = None
        with torch.no_grad():
            for x, _, _ in train_loader:
                x = x.to(self.device)
                phi = self.net(x)
                if c is None:
                    c = phi.sum(0)
                else:
                    c += phi.sum(0)
                n += phi.size(0)
        c /= n
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.c = c.detach().clone()
        # --- OPTIONAL WARNING in _initialize_center_c (add inside method before return) ---
        logging.getLogger(__name__).info("Center c initialized. NOTE: train loader may contain anomalies if labels=1 present.")

    def anomaly_score(self, x: torch.Tensor):
        """
        Return squared distance ||phi(x) - c||^2 (higher = more anomalous).
        """
        assert self.net is not None and self.c is not None, "Train SVDD (and initialize center) first."
        self.net.eval()
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.net(x)
            d = torch.sum((phi - self.c.to(self.device)) ** 2, dim=1)
        return d

    def test_svdd(self, test_loader):
        """
        Compute anomaly scores (and optional AUC if labels exist).
        """
        from sklearn.metrics import roc_auc_score
        scores_list, labels_list = [], []
        self.net.eval()
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(self.device)
                s = self.anomaly_score(x)
                scores_list.append(s.cpu())
                labels_list.append(y)
        scores = torch.cat(scores_list).numpy()
        labels = torch.cat(labels_list).numpy()
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = None
        self.results['test_scores'] = scores.tolist()
        self.results['test_auc'] = auc
        return scores, labels, auc

# --- OPTIONAL WARNING in _initialize_center_c (add inside method before return) ---
        # After computing self.c:
        # (Add)
        # logging.getLogger(__name__).info("Center c initialized. NOTE: train loader may contain anomalies if labels=1 present.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    DATA_PATH = r"D:\Academic\QML Intern\Anomaly Detection\Fraud Detection\creditcard_data.csv"

    # Load data (already defined earlier once; safe if re-run)
    train_loader, test_loader = get_fraud_dataloaders(data_path=DATA_PATH, batch_size=128)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')

    svdd_model = QSVDD()
    svdd_model.device = device
    
    # 1. Pretrain autoencoder
    svdd_model.pretrain(
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=10,         # adjust as needed
        lr=0.05,
        device=device
    )

    # 2. Build main (feature) network; use PCA feature dimension (infer from a batch)
    sample_batch = next(iter(train_loader))[0]
    feature_dim = sample_batch.shape[1]
    # Rebuild a feature network matching flat feature_dim; reuse existing ParametricHybridNet by setting input_shape
    svdd_model.build_main_network(
        classical_dims=[64, 32],
        n_qubits=n_qubits,
        q_depth=q_depth,
        activation='relu',
        input_shape=(feature_dim,)  # treat features as a flat vector
    )

    # 3. Transfer pretrained quantum encoder weights and freeze for first epochs
    svdd_model.transfer_pretrained_encoder(freeze=True)

    # 4. Train Deep SVDD (unfreeze quantum weights after 3 warmup epochs)
    svdd_model.train_svdd(
        train_loader=train_loader,
        lr=1e-3,
        n_epochs=15,
        freeze_quantum_epochs=3
    )

    # 5. Evaluate / compute anomaly scores + AUC
    scores, labels, auc = svdd_model.test_svdd(test_loader)
    logging.info(f"AUC: {auc}")
    logging.info(f"First 5 anomaly scores: {scores[:5]}")