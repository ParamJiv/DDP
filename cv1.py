# %% [markdown]
# ## 1. Import Required Libraries
# (Keep previous imports, add Keras layers)

# %%
# Quantum Computing Libraries
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.ops import Dgate, BSgate, Kgate, Sgate, Rgate

# Deep Learning and Numerical Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
# import scipy as sp # Not strictly needed for this version

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D # Not used here

# Sklearn Libraries (Keep only what's needed for metrics/splitting if necessary)
from sklearn.preprocessing import StandardScaler # Still useful for PCA comparison or other data
from sklearn.decomposition import PCA # Keep for comparison if desired
from sklearn.model_selection import train_test_split # If splitting differently

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# System and Warning Handling
import os
import warnings
from tqdm import tqdm # For progress bars
import time

# Datasets
from tensorflow.keras.datasets import mnist

# Configure TensorFlow and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True) # Keep eager execution for easier debugging with SF

print("TensorFlow version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# Ensure NumPy version compatibility if needed
try:
    np_major_version = int(np.__version__.split('.')[0])
    if np_major_version >= 2:
        print(f"Warning: NumPy version {np.__version__} >= 2.0.0 detected. "
              "Ensure TensorFlow and Strawberry Fields are compatible or downgrade NumPy ('pip install numpy<2').")
except Exception as e:
    print(f"Could not check NumPy version: {e}")


# %% [markdown]
# ## 2. Load and Prepare MNIST Dataset (Binary 0 vs 1)
# Load MNIST, filter for digits 0 and 1, reshape for CNN, and normalize.

# %%
# Load MNIST dataset
(x_train_all, y_train_all), (x_test_all, y_test_all) = mnist.load_data()
print(f"Original MNIST training shape: {x_train_all.shape}, labels: {y_train_all.shape}")
print(f"Original MNIST test shape: {x_test_all.shape}, labels: {y_test_all.shape}")

# --- Filter for digits 0 and 1 ---
def filter_digits(x, y, digit1, digit2):
    keep = (y == digit1) | (y == digit2)
    x_filtered = x[keep]
    y_filtered = y[keep]
    # Remap labels: digit1 -> 0, digit2 -> 1
    y_mapped = np.where(y_filtered == digit1, 0, 1)
    return x_filtered, y_mapped

digit1, digit2 = 0, 1 # Choose digits to classify
x_train, y_train = filter_digits(x_train_all, y_train_all, digit1, digit2)
x_test, y_test = filter_digits(x_test_all, y_test_all, digit1, digit2)

print(f"\nFiltered MNIST training shape ('{digit1}' vs '{digit2}'): {x_train.shape}, labels: {y_train.shape}")
print(f"Filtered MNIST test shape ('{digit1}' vs '{digit2}'): {x_test.shape}, labels: {y_test.shape}")
print(f"Training label counts: {np.bincount(y_train)}")
print(f"Test label counts: {np.bincount(y_test)}")

# --- Preprocessing for CNN ---
# Add channel dimension (MNIST is grayscale)
x_train_cnn = x_train[..., np.newaxis]
x_test_cnn = x_test[..., np.newaxis]
print(f"\nShapes for CNN: train={x_train_cnn.shape}, test={x_test_cnn.shape}")

# Normalize pixel values (0-1)
x_train_norm = x_train_cnn.astype('float32') / 255.0
x_test_norm = x_test_cnn.astype('float32') / 255.0

# Keep labels as 0 or 1 (NumPy arrays)
y_train_np = y_train.astype(int)
y_test_np = y_test.astype(int)

# --- (Optional) Shuffle Training Data ---
# It's often good practice, although Keras can shuffle during training too.
# We'll rely on shuffling within the training loop later.
X_shuffled = x_train_norm
y_shuffled = y_train_np

# %%
# Visualize some sample images
plt.figure(figsize=(10, 4))
num_samples_to_show = 10
if len(X_shuffled) >= num_samples_to_show:
    indices = np.random.choice(len(X_shuffled), num_samples_to_show, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_shuffled[idx].squeeze(), cmap='gray') # Use squeeze to remove channel dim for imshow
        plt.title(f"Label: {y_shuffled[idx]}")
        plt.axis('off')
    plt.suptitle(f"Sample MNIST Images ('{digit1}' vs '{digit2}')")
    plt.show()
else:
    print("Not enough samples to display.")


# %% [markdown]
# ## 3. Configure Quantum Parameters
# (Keep most parameters, adjust input size based on CNN output if needed, but here we output fixed 14)

# %%
# Quantum Circuit Parameters
mode_number = 2          # Number of photonic modes
depth = 6                # Number of quantum layers
cutoff = 10              # Fock basis truncation
batch_size = 32         # Training batch size (adjust based on memory)

# Gate Parameter Initialization
sdev_photon = 0.1       # Standard deviation for photonic parameters
sdev = 1                # Standard deviation for general parameters

# Parameter clipping values
disp_clip = 5           # Displacement gate clipping
sq_clip = 5             # Squeezing gate clipping
kerr_clip = 1           # Kerr gate clipping

# Classical Network Output (for Quantum Encoding)
output_neurons = 14     # Fixed size for the quantum input layer

print("Quantum Circuit Configuration:")
print(f"  Mode number: {mode_number}")
print(f"  Depth: {depth}")
print(f"  Cutoff: {cutoff}")
print(f"  Batch size: {batch_size}")
print(f"\nClassical Network Output Size (for Quantum): {output_neurons}")

# %% [markdown]
# ## 4. Define Classical CNN Architecture (using Keras Functional API)
# Create the CNN feature extractor followed by a dense layer for quantum parameters.

# %%
def create_classical_cnn_model(input_shape, num_quantum_params):
    """Creates the classical CNN part of the hybrid model."""
    inputs = Input(shape=input_shape, name="image_input")

    # Convolutional layers
    x = Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Output shape: (None, 7, 7, 16) for MNIST 28x28 input

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x) # Intermediate dense layer
    quantum_params_output = Dense(num_quantum_params, activation="linear", name="quantum_params")(x) # Output layer for quantum params

    model = Model(inputs=inputs, outputs=quantum_params_output, name="classical_cnn_feature_extractor")
    return model

# Define input shape for MNIST (height, width, channels)
cnn_input_shape = X_shuffled.shape[1:] # Should be (28, 28, 1)

# Create the classical model
classical_model = create_classical_cnn_model(cnn_input_shape, output_neurons)
classical_model.summary()

# %% [markdown]
# ## 5. Define Quantum Circuit Functions & Initialize Quantum Variables
# (Keep qnn_var, input_qnn_layer, qnn_layer functions exactly as before)

# %%
# --- qnn_var function (from original code) ---
def qnn_var():
    bs_in_interferometer = int(1.0 * mode_number * (mode_number - 1) / 2)
    bs_variables = tf.Variable(tf.random.normal(shape=[depth, bs_in_interferometer, 2, 2], stddev=sdev), name="bs_variables")
    phase_variables = tf.Variable(tf.random.normal(shape=[depth, mode_number, 2], stddev=sdev), name="phase_variables")
    sq_magnitude_variables = tf.Variable(tf.random.normal(shape=[depth, mode_number], stddev=sdev_photon), name="sq_magnitude")
    sq_phase_variables = tf.Variable(tf.random.normal(shape=[depth, mode_number], stddev=sdev), name="sq_phase")
    disp_magnitude_variables = tf.Variable(tf.random.normal(shape=[depth, mode_number], stddev=sdev_photon), name="disp_magnitude")
    disp_phase_variables = tf.Variable(tf.random.normal(shape=[depth, mode_number], stddev=sdev), name="disp_phase")
    kerr_variables = tf.Variable(tf.random.normal(shape=[depth, mode_number], stddev=sdev_photon), name="kerr_variables")
    parameters_dict = {
        'bs_variables': bs_variables, 'phase_variables': phase_variables,
        'sq_magnitude_variables': sq_magnitude_variables, 'sq_phase_variables': sq_phase_variables,
        'disp_magnitude_variables': disp_magnitude_variables, 'disp_phase_variables': disp_phase_variables,
        'kerr_variables': kerr_variables,
    }
    return parameters_dict

# --- input_qnn_layer function (from original code) ---
def input_qnn_layer(parameters_dict, q):
    output_layer = parameters_dict['output_layer'] # This now comes from the Keras model
    Sgate(tf.clip_by_value(output_layer[:, 0], -sq_clip, sq_clip), output_layer[:, 1]) | q[0]
    Sgate(tf.clip_by_value(output_layer[:, 2], -sq_clip, sq_clip), output_layer[:, 3]) | q[1]
    BSgate(output_layer[:, 4], output_layer[:, 5]) | (q[0], q[1])
    Rgate(output_layer[:, 6]) | q[0]
    Rgate(output_layer[:, 7]) | q[1]
    Dgate(tf.clip_by_value(output_layer[:, 8], -disp_clip, disp_clip), output_layer[:, 9]) | q[0]
    Dgate(tf.clip_by_value(output_layer[:, 10], -disp_clip, disp_clip), output_layer[:, 11]) | q[1]
    Kgate(tf.clip_by_value(output_layer[:, 12], -kerr_clip, kerr_clip)) | q[0]
    Kgate(tf.clip_by_value(output_layer[:, 13], -kerr_clip, kerr_clip)) | q[1]

# --- qnn_layer function (from original code) ---
def qnn_layer(parameters_dict, layer_number, q):
    bs_variables = parameters_dict['bs_variables']
    phase_variables = parameters_dict['phase_variables']
    sq_magnitude_variables = parameters_dict['sq_magnitude_variables']
    sq_phase_variables = parameters_dict['sq_phase_variables']
    disp_magnitude_variables = parameters_dict['disp_magnitude_variables']
    disp_phase_variables = parameters_dict['disp_phase_variables']
    kerr_variables = parameters_dict['kerr_variables']
    BSgate(bs_variables[layer_number, 0, 0, 0], bs_variables[layer_number, 0, 0, 1]) | (q[0], q[1])
    for i in range(mode_number): Rgate(phase_variables[layer_number, i, 0]) | q[i]
    for i in range(mode_number): Sgate(tf.clip_by_value(sq_magnitude_variables[layer_number, i], -sq_clip, sq_clip), sq_phase_variables[layer_number, i]) | q[i]
    BSgate(bs_variables[layer_number, 0, 1, 0], bs_variables[layer_number, 0, 1, 1]) | (q[0], q[1])
    for i in range(mode_number): Rgate(phase_variables[layer_number, i, 1]) | q[i]
    for i in range(mode_number): Dgate(tf.clip_by_value(disp_magnitude_variables[layer_number, i], -disp_clip, disp_clip), disp_phase_variables[layer_number, i]) | q[i]
    for i in range(mode_number): Kgate(tf.clip_by_value(kerr_variables[layer_number, i], -kerr_clip, kerr_clip)) | q[i]


# --- Initialize Quantum Variables ---
print("Initializing quantum variables...")
qnn_params = qnn_var()
bs_variables = qnn_params['bs_variables']
phase_variables = qnn_params['phase_variables']
sq_magnitude_variables = qnn_params['sq_magnitude_variables']
sq_phase_variables = qnn_params['sq_phase_variables']
disp_magnitude_variables = qnn_params['disp_magnitude_variables']
disp_phase_variables = qnn_params['disp_phase_variables']
kerr_variables = qnn_params['kerr_variables']
print("Quantum variables initialized.")

# Combine all trainable variables (Classical Keras model + Quantum TF Variables)
all_trainable_params = classical_model.trainable_variables + list(qnn_params.values())
print(f"Total number of parameter groups to train: {len(all_trainable_params)}")

# %% [markdown]
# ## 6. Implement Quantum Loss and Training Functions
# (Keep compute_quantum_loss adapted for 0/1 labels, modify train function for Keras model)

# %%
# --- compute_quantum_loss function (ADAPTED for 0/1 labels) ---
def compute_quantum_loss(output_layer_params, batch_y_labels, depth, mode_number, cutoff, actual_batch_size):
    """Compute loss using quantum circuit measurement. Expects batch_y_labels as 0 or 1."""
    parameters_dict = {
        'output_layer': output_layer_params, # Renamed for clarity
        **qnn_params # Include global quantum vars directly
    }
    eng1 = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": actual_batch_size})
    q1 = sf.Program(mode_number)
    with q1.context as q:
        input_qnn_layer(parameters_dict=parameters_dict, q=q)
        for j in range(depth):
            qnn_layer(parameters_dict=parameters_dict, layer_number=j, q=q)
    state = eng1.run(q1)
    ket1 = state.state.ket()

    classification_labels = tf.cast(batch_y_labels, tf.int32) # Ensure labels are int
    loss = 0.0
    for i in range(actual_batch_size):
        # Map label 0 to |1,0> and label 1 to |0,1>
        if classification_labels[i] == 0:
            target_prob = tf.abs(ket1[i, 1, 0]) ** 2 # Target |1,0>
        else:
            target_prob = tf.abs(ket1[i, 0, 1]) ** 2 # Target |0,1>
        loss += (1 - target_prob) ** 2
    return loss / actual_batch_size

# --- train_quantum_model function (MODIFIED for Keras classical model) ---
def train_hybrid_model_cnn(input_images, target_labels_np, classical_model, qnn_params,
                           depth, cutoff, batch_size, mode_number,
                           num_epochs=10, learning_rate=0.01, verbose=True):
    """Main training function for the hybrid CNN-Quantum model."""

    # Define optimizer (use Keras optimizer)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Convert data to tensors
    input_data_tf = tf.convert_to_tensor(input_images, dtype=tf.float32)
    target_labels_tf = tf.convert_to_tensor(target_labels_np, dtype=tf.int32) # Use int labels (0 or 1)

    num_samples = input_data_tf.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size # Handle partial batches

    print(f"Training Configuration:")
    print(f"  Samples: {num_samples}")
    print(f"  Batches per epoch: {num_batches}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")

    loss_history = []
    start_time = time.time()

    # Get all trainable parameters (Keras classical + TF quantum)
    all_trainable_params = classical_model.trainable_variables + list(qnn_params.values())

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        indices = tf.random.shuffle(tf.range(num_samples))
        input_shuffled_tf = tf.gather(input_data_tf, indices)
        labels_shuffled_tf = tf.gather(target_labels_tf, indices)

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs}", unit="batch")

        for i in pbar:
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch_x = input_shuffled_tf[start:end]
            batch_y = labels_shuffled_tf[start:end] # Labels are 0 or 1
            actual_batch_size = batch_x.shape[0]

            if actual_batch_size == 0: continue

            with tf.GradientTape() as tape:
                # Forward pass through classical Keras model to get quantum params
                output_layer_params = classical_model(batch_x, training=True)

                # Compute quantum loss (pass the 14 params and 0/1 labels)
                loss = compute_quantum_loss(output_layer_params, batch_y, depth, mode_number,
                                            cutoff, actual_batch_size)

            # Compute gradients for ALL trainable parameters
            grads = tape.gradient(loss, all_trainable_params)

            # Check for None gradients (can happen with disconnected graphs or issues)
            filtered_grads_params = []
            for grad, param in zip(grads, all_trainable_params):
                if grad is not None:
                    # Optional: Clip gradients
                    grad = tf.clip_by_norm(grad, 1.0)
                    filtered_grads_params.append((grad, param))
                else:
                    # Be cautious if essential parameters have None gradients
                    print(f"Warning: None gradient for parameter: {param.name}")

            if not filtered_grads_params:
                print(f"Warning: No valid gradients found at epoch {epoch}, batch {i}. Skipping update.")
                continue

            # Apply gradients
            optimizer.apply_gradients(filtered_grads_params)
            current_loss = loss.numpy()
            epoch_loss += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        loss_history.append(avg_loss)

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch:02d} - Avg Loss: {avg_loss:.6f} - Time: {elapsed_time:.2f}s")

    print(f"Total training time: {time.time() - start_time:.2f}s")
    return classical_model, qnn_params, loss_history # Return updated models/params

print("Quantum loss and Training functions updated for CNN.")


# %% [markdown]
# ## 7. Train the Hybrid Model

# %%
# Initialize quantum variables globally (if not already done)
if 'qnn_params' not in globals():
    qnn_params = qnn_var()
    bs_variables = qnn_params['bs_variables']
    phase_variables = qnn_params['phase_variables']
    sq_magnitude_variables = qnn_params['sq_magnitude_variables']
    sq_phase_variables = qnn_params['sq_phase_variables']
    disp_magnitude_variables = qnn_params['disp_magnitude_variables']
    disp_phase_variables = qnn_params['disp_phase_variables']
    kerr_variables = qnn_params['kerr_variables']
    print("Re-initialized quantum variables.")


# Train the hybrid model
print("\nStarting hybrid CNN-Quantum model training...")
print("=" * 50)

# Make sure to pass the correct data: CNN inputs (X_shuffled) and 0/1 labels (y_shuffled)
trained_classical_model, trained_qnn_params, loss_history = train_hybrid_model_cnn(
    input_images=X_shuffled,
    target_labels_np=y_shuffled,
    classical_model=classical_model, # Pass the Keras model instance
    qnn_params=qnn_params, # Pass the dictionary of TF quantum variables
    depth=depth,
    cutoff=cutoff,
    batch_size=batch_size,
    mode_number=mode_number,
    num_epochs=5, # Start with fewer epochs for testing
    learning_rate=0.005, # Adjust learning rate
    verbose=True
)

print("=" * 50)
print("Training completed!")

# Update global qnn_params with trained values for prediction
qnn_params = trained_qnn_params


# %%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=2, marker='o', markersize=4)
plt.title('Hybrid CNN-Quantum Model Training Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

if loss_history:
    print(f"Initial training loss: {loss_history[0]:.6f}")
    print(f"Final training loss: {loss_history[-1]:.6f}")
    if loss_history[0] > 0:
      reduction = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
      print(f"Loss reduction: {reduction:.2f}%")
else:
    print("No loss history recorded.")


# %% [markdown]
# ## 8. Make Predictions and Evaluate Performance
# (Adapt prediction function for Keras model)

# %%
# --- predict_hybrid_model_cnn function (MODIFIED) ---
def predict_hybrid_model_cnn(input_images, classical_model, qnn_params_pred, # Use specific params for prediction
                               depth, mode_number, cutoff, batch_size_pred=128): # Larger batch for prediction
    """Make predictions using the trained hybrid CNN-Quantum model."""

    # Ensure qnn_params_pred contains TF Variables for the quantum circuit
    # These should be the *trained* variables returned by the training function

    input_data_tf = tf.convert_to_tensor(input_images, dtype=tf.float32)
    num_samples = input_data_tf.shape[0]
    predictions = []
    probabilities = [] # Stores [P(0), P(1)]

    num_batches = (num_samples + batch_size_pred - 1) // batch_size_pred
    print(f"\n🔮 Predicting on {num_samples} samples using {num_batches} batches (batch size {batch_size_pred})...")

    pbar_pred = tqdm(range(0, num_samples, batch_size_pred),
                     desc="🧠 Hybrid CNN-Q Prediction", unit="batch", total=num_batches)

    # Prepare quantum parameters dictionary ONCE for prediction
    parameters_dict_pred = {
        # 'output_layer' will be updated per batch
        **qnn_params_pred # Use the trained quantum variables
    }

    for start in pbar_pred:
        end = min(start + batch_size_pred, num_samples)
        batch_x = input_data_tf[start:end]
        actual_batch_size_pred = batch_x.shape[0]

        if actual_batch_size_pred == 0: continue

        # --- Classical Part ---
        # Get quantum parameters from the classical Keras model
        output_layer_params_batch = classical_model(batch_x, training=False)
        parameters_dict_pred['output_layer'] = output_layer_params_batch # Update the dict

        # --- Quantum Part ---
        # Run quantum circuit simulation for the batch
        eng_pred = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": actual_batch_size_pred})
        q_pred = sf.Program(mode_number)
        with q_pred.context as q:
            input_qnn_layer(parameters_dict=parameters_dict_pred, q=q)
            for j in range(depth):
                qnn_layer(parameters_dict=parameters_dict_pred, layer_number=j, q=q)

        state_pred = eng_pred.run(q_pred)
        ket_batch = state_pred.state.ket() # Shape: (batch_size, cutoff, cutoff)

        # Process results for each sample in the batch
        for i in range(actual_batch_size_pred):
            prob_state_10 = tf.abs(ket_batch[i, 1, 0]) ** 2 # Corresponds to label 0
            prob_state_01 = tf.abs(ket_batch[i, 0, 1]) ** 2 # Corresponds to label 1

            # Normalize probabilities (optional but good practice)
            # total_prob = prob_state_10 + prob_state_01
            # if total_prob > 1e-6: # Avoid division by zero
            #     prob_0_norm = prob_state_10 / total_prob
            #     prob_1_norm = prob_state_01 / total_prob
            # else:
            #     prob_0_norm, prob_1_norm = 0.5, 0.5 # Default if probs are tiny

            # Prediction based on higher probability
            prediction = 0 if prob_state_10 > prob_state_01 else 1
            predictions.append(prediction)

            # Store NON-NORMALIZED probabilities as [P(0)=P(|1,0>), P(1)=P(|0,1>)] for consistency
            probabilities.append([float(prob_state_10), float(prob_state_01)])

        pbar_pred.set_postfix({"samples": f"{end}/{num_samples}"})

    pbar_pred.close()
    return np.array(predictions), np.array(probabilities)


print("Prediction function updated.")

# %%
# --- Make Predictions ---
print("\nMaking predictions with trained hybrid CNN-Quantum model...")

# Use the correct data: normalized test images (x_test_norm)
# Pass the TRAINED classical model and TRAINED quantum parameters
test_predictions_cnn, test_probabilities_cnn = predict_hybrid_model_cnn(
    input_images=x_test_norm,
    classical_model=trained_classical_model, # Use the trained Keras model
    qnn_params_pred=trained_qnn_params,     # Use the trained quantum TF Variables
    depth=depth,
    mode_number=mode_number,
    cutoff=cutoff
)

print("\nPredictions completed.")
# Optional: Predict on training data too for comparison
# train_predictions_cnn, train_probabilities_cnn = predict_hybrid_model_cnn(...)


# %%
# --- Evaluate Performance ---
print("\n--- Hybrid CNN-Quantum Model Evaluation (Test Data) ---")

# Define the evaluation function (if not already defined)
def evaluate_performance(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError: # Handle cases with only one class predicted
        roc_auc = 0.0
        print("Warning: ROC AUC cannot be calculated (likely only one class predicted).")

    print(f"\n=== {dataset_name} Results === ")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (Class 1): {precision:.4f}") # Binary precision often refers to class 1
    print(f"Recall (Class 1):    {recall:.4f}")
    print(f"F1-Score (Class 1):  {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix (Rows: True, Cols: Pred):")
    print(cm)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[f"Digit {digit1} (0)", f"Digit {digit2} (1)"], zero_division=0))

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'roc_auc': roc_auc, 'confusion_matrix': cm}

# Evaluate test performance using the correct test labels (y_test_np)
test_metrics_cnn = evaluate_performance(y_test_np, test_predictions_cnn, "Test Data (CNN-Quantum)")

# %%
# Plot Confusion Matrix for Test Data
plt.figure(figsize=(6, 5))
cm_test_cnn = test_metrics_cnn['confusion_matrix']
sns.heatmap(cm_test_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Digit {digit1}", f"Digit {digit2}"],
            yticklabels=[f"Digit {digit1}", f"Digit {digit2}"])
plt.title('Test Data Confusion Matrix (CNN-Quantum)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %%
# Plot ROC Curve for Test Data
if test_metrics_cnn['roc_auc'] > 0: # Check if calculable
    # Use probability of the positive class (label 1, which is index 1 in probabilities array)
    fpr, tpr, thresholds = roc_curve(y_test_np, test_probabilities_cnn[:, 1])
    auc_score = roc_auc_score(y_test_np, test_probabilities_cnn[:, 1])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Data ROC Curve (CNN-Quantum)')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("\nSkipping ROC curve plotting as AUC was not calculable.")

# %%
# Plot probability distributions for the test set
plt.figure(figsize=(8, 6))
# Probabilities of predicting class 1 (Digit digit2)
prob_class_1_true_0 = test_probabilities_cnn[y_test_np == 0, 1]
prob_class_1_true_1 = test_probabilities_cnn[y_test_np == 1, 1]