"""
Ship a Model: From a Trained PyTorch Network to a Monitored Service

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - set_seed
import numpy as np

FEATURES = ["distance_km", "items", "hour", "driver_rating", "traffic_index", "rain"]


def load_deliveries(n=2400, seed=11, winter=0.0):
    """Past deliveries: six numbers per order, and 1 if it arrived late, 0 if on time.

    winter between 0 and 1 makes traffic heavier and rain more common, and rain matters more,
    the way it does in a winter month.
    """
    rs = np.random.RandomState(seed)
    distance = rs.gamma(2.0, 2.5, size=n)
    items = rs.randint(1, 12, size=n).astype(float)
    hour = rs.randint(8, 23, size=n).astype(float)
    rating = np.clip(rs.normal(4.6, 0.3, size=n), 1.0, 5.0)
    traffic = np.clip(rs.normal(0.4 + 0.3 * winter, 0.2, size=n), 0.0, 2.0)
    rain = (rs.rand(n) < 0.15 + 0.35 * winter).astype(float)
    rush = ((hour >= 17) & (hour <= 19)).astype(float)
    z = (-3.4 + 0.28 * distance + 0.06 * items + 1.4 * rush + 2.0 * traffic
         + (0.6 + 2.4 * winter) * rain - 1.2 * (rating - 4.6))
    late = (rs.rand(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    X = np.column_stack([distance, items, hour, rating, traffic, rain]).astype(np.float32)
    return X, late
# ---- end provided


import random
import torch


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch random number generators."""

    # Set the random seed for Python's built-in random module.
    random.seed(seed)

    # Set the random seed for NumPy's global random number generator.
    np.random.seed(seed)

    # Set the random seed for PyTorch (CPU and CUDA devices).
    torch.manual_seed(seed)

# Step 2 - make_loader
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size=64, seed=0, shuffle=True):
    """Create a reproducible DataLoader over float32 features and int64 labels."""

    # Convert features and labels into tensors with the required data types.
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.int64)

    # Combine features and labels into a single dataset.
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create a dedicated random generator so that data shuffling
    # does not depend on PyTorch's global random-number generator.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Return a DataLoader using the dedicated seeded generator.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )

    return loader

# Step 3 - DeliveryNet
from torch import nn


class DeliveryNet(nn.Module):
    """MLP that standardizes raw inputs with stored mean/std buffers and returns 2 logits."""

    def __init__(self, mean, std, hidden=32):
        super().__init__()

        # Store independent float32 copies of the preprocessing parameters.
        # Registered buffers are saved in state_dict() but are not trainable parameters.
        self.register_buffer(
            "mean",
            torch.as_tensor(mean, dtype=torch.float32).clone()
        )

        self.register_buffer(
            "std",
            torch.as_tensor(std, dtype=torch.float32).clone()
        )

        # Determine the number of input features from the mean vector.
        n_features = len(self.mean)

        # Define the neural network architecture.
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        # Standardize raw input features using the stored mean and standard deviation.
        x = (x - self.mean) / self.std

        # Pass the standardized features through the neural network.
        return self.layers(x)

# Step 4 - train_model
def train_model(X, y, seed=0, epochs=8, lr=1e-2, batch_size=64, hidden=32):
    """Seeded training of a DeliveryNet; returns (model, per-epoch mean losses)."""

    # Step 1: Seed Python, NumPy, and PyTorch random number generators.
    set_seed(seed)

    # Step 2: Convert the input features to a float32 PyTorch tensor.
    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    # Calculate the mean and population standard deviation of each feature.
    mean = X_tensor.mean(dim=0)
    std = X_tensor.std(dim=0, unbiased=False)

    # Prevent division by zero or extremely small standard deviations.
    std = std.clamp_min(1e-6)

    # Step 3: Initialize the model with its preprocessing parameters.
    model = DeliveryNet(mean, std, hidden=hidden)

    # Step 4: Create a reproducible DataLoader using the specified seed.
    loader = make_loader(
        X,
        y,
        batch_size=batch_size,
        seed=seed,
        shuffle=True
    )

    # Define the Adam optimizer and cross-entropy loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Store the average training loss for each epoch.
    epoch_losses = []

    # Step 5: Train the model for the specified number of epochs.
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0

        for batch_X, batch_y in loader:

            # Clear gradients from the previous optimization step.
            optimizer.zero_grad()

            # Forward pass: obtain the model's two-class logits.
            logits = model(batch_X)

            # Calculate cross-entropy loss for the current batch.
            loss = criterion(logits, batch_y)

            # Backward pass: calculate gradients.
            loss.backward()

            # Update model parameters using Adam.
            optimizer.step()

            # Accumulate loss weighted by the number of examples.
            batch_examples = batch_X.size(0)
            total_loss += loss.item() * batch_examples
            total_examples += batch_examples

        # Calculate the average loss across all examples in this epoch.
        average_loss = total_loss / total_examples
        epoch_losses.append(average_loss)

    # Return the trained model and per-epoch average training losses.
    return model, epoch_losses

# Step 5 - predict_proba
def predict_proba(model, X):
    """Probability of class 1 for each row, as a NumPy array, in eval mode without gradients."""

    # Switch the model to evaluation mode.
    model.eval()

    # Get the device on which the model is currently stored.
    device = next(model.parameters()).device

    # Convert input data into a float32 tensor on the model's device.
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)

    # Support a single feature vector as well as a batch of feature vectors.
    if X_tensor.ndim == 1:
        X_tensor = X_tensor.unsqueeze(0)

    # Disable gradient tracking during inference.
    with torch.no_grad():

        # Obtain the model's raw classification logits.
        logits = model(X_tensor)

        # Convert logits into probabilities and select class 1 (late delivery).
        probabilities = torch.softmax(logits, dim=1)[:, 1]

    # Move predictions to the CPU and return them as a 1-D NumPy array.
    return probabilities.cpu().numpy()

# Step 6 - validate_request
def validate_request(payload):
    """Return (row, []) for a valid request, or (None, errors) otherwise."""

    # Define the required fields, their valid ranges, and whether they
    # must contain whole-number values.
    field_rules = {
        "distance_km": (0, 100, False),
        "items": (1, 50, True),
        "hour": (0, 23, True),
        "driver_rating": (1, 5, False),
        "traffic_index": (0, 2, False),
        "rain": (0, 1, True),
    }

    errors = []
    row = []

    # Validate each required field in the specified order.
    for field, (minimum, maximum, must_be_whole) in field_rules.items():

        # Check whether the field is missing or explicitly set to None.
        if field not in payload or payload[field] is None:
            errors.append(f"missing: {field}")
            continue

        value = payload[field]

        # Reject booleans, even though Python treats bool as a subclass of int.
        # Accept standard Python and NumPy integer and floating-point values.
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            errors.append(f"not a number: {field}")
            continue

        # Convert the numeric value to a Python float.
        numeric_value = float(value)

        # Items, hour, and rain must contain whole-number values.
        # Values such as 3 and 3.0 are both accepted.
        if must_be_whole and not numeric_value.is_integer():
            errors.append(f"not a whole number: {field}")
            continue

        # Check the inclusive lower and upper bounds.
        if not minimum <= numeric_value <= maximum:
            errors.append(f"out of range: {field}")
            continue

        # Append the validated value as a float.
        row.append(numeric_value)

    # Report every unexpected field, sorted alphabetically by name.
    unknown_fields = sorted(
        (key for key in payload if key not in field_rules),
        key=str
    )

    for name in unknown_fields:
        errors.append(f"unknown field: {name}")

    # Return the validated feature row only if there are no errors.
    if errors:
        return None, errors

    return row, []

# Step 7 - save_artifact
import io


def save_artifact(model, metadata):
    """Serialize the model state dict, hidden width, and metadata into bytes."""

    # Create an in-memory binary buffer.
    buffer = io.BytesIO()

    # Package the model information into a dictionary.
    artifact = {
        # Include model weights and registered preprocessing buffers.
        "state_dict": model.state_dict(),

        # Store the width of the first hidden layer.
        "hidden": model.layers[0].out_features,

        # Store an independent dictionary copy of the metadata.
        "metadata": dict(metadata)
    }

    # Serialize the artifact into the in-memory buffer.
    torch.save(artifact, buffer)

    # Return the serialized artifact as bytes.
    return buffer.getvalue()

# Step 8 - load_artifact
def load_artifact(blob):
    """Rebuild a DeliveryNet from save_artifact bytes; return (model in eval mode, metadata)."""

    # Wrap the serialized bytes in an in-memory binary buffer.
    buffer = io.BytesIO(blob)

    # Safely load the serialized artifact and map tensors to the CPU.
    artifact = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True
    )

    # Extract the saved model state dictionary.
    state_dict = artifact["state_dict"]

    # Retrieve the preprocessing parameters stored as model buffers.
    mean = state_dict["mean"]
    std = state_dict["std"]

    # Retrieve the hidden layer width used by the original model.
    hidden = artifact["hidden"]

    # Reconstruct the DeliveryNet with the original architecture
    # and preprocessing parameters.
    model = DeliveryNet(
        mean=mean,
        std=std,
        hidden=hidden
    )

    # Restore the original model parameters and registered buffers.
    model.load_state_dict(state_dict)

    # Set the reconstructed model to evaluation mode.
    model.eval()

    # Return the reconstructed model and a copy of its metadata.
    metadata = dict(artifact["metadata"])

    return model, metadata

# Step 9 - export_model
def export_model(model, example_input):
    """Trace the model in eval mode and return TorchScript bytes."""

    # Switch the model to evaluation mode before tracing.
    model.eval()

    # Disable gradient tracking during the tracing process.
    with torch.no_grad():

        # Trace the model using the provided example input.
        # Tracing records the operations performed by the model.
        traced_model = torch.jit.trace(
            model,
            example_input
        )

    # Create an in-memory binary buffer for the TorchScript model.
    buffer = io.BytesIO()

    # Serialize the traced model into the buffer.
    torch.jit.save(traced_model, buffer)

    # Return the serialized TorchScript model as bytes.
    return buffer.getvalue()

# Step 10 - batch_predict (not yet solved)
# TODO: implement

# Step 11 - quantize_weights (not yet solved)
# TODO: implement

# Step 12 - drift_report (not yet solved)
# TODO: implement

# Step 13 - shadow_compare (not yet solved)
# TODO: implement

