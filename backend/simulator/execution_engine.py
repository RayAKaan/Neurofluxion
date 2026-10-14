"""
Neurofluxion's standalone NumPy execution engine.

The /api/execute routes use this engine independently from the inspectable
NetworkGraph simulator. This module deliberately implements a dense feed-forward
subset; unsupported layer types are rejected rather than silently simulated.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class LayerData:
    layer_index: int
    layer_type: str
    pre_activation: Optional[np.ndarray] = None
    post_activation: Optional[np.ndarray] = None
    gradient_weights: Optional[np.ndarray] = None
    gradient_bias: Optional[np.ndarray] = None
    deltas: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    biases: Optional[np.ndarray] = None
    activation_stats: Dict[str, float] = field(default_factory=dict)
    gradient_stats: Dict[str, float] = field(default_factory=dict)
    weight_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class ForwardResult:
    layers: List[LayerData]
    final_output: np.ndarray
    total_params: int
    execution_time_ms: float
    device_used: str


@dataclass
class BackwardResult:
    layers: List[LayerData]
    loss_value: float
    total_gradient_norm: float
    execution_time_ms: float


@dataclass
class TrainingResult:
    loss_history: List[float]
    metrics_history: List[Dict[str, float]]
    final_metrics: Dict[str, float]
    total_epochs: int
    execution_time_ms: float


class ExecutionEngine:
    """Dense feed-forward execution with numerical forward/backward/training."""

    SUPPORTED_LAYERS = {"input", "dense", "output"}
    SUPPORTED_ACTIVATIONS = {"linear", "relu", "leaky_relu", "sigmoid", "tanh", "softmax"}
    SUPPORTED_LOSSES = {"mse", "bce"}

    def __init__(self, device: str = "auto"):
        self._device_preference = device
        self._backend = "cpu"
        self._graph: Optional[Dict] = None
        self._last_forward: Optional[Dict[str, Any]] = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_gpu_available(self) -> bool:
        return CUPY_AVAILABLE

    def _get_backend(self) -> str:
        if self._device_preference == "gpu" and CUPY_AVAILABLE:
            return "cupy"
        return "numpy"

    def _xp(self):
        return cp if self._backend == "cupy" and CUPY_AVAILABLE else np

    def _to_numpy(self, arr) -> np.ndarray:
        if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _compute_stats(self, arr: np.ndarray) -> Dict[str, float]:
        value = self._to_numpy(arr)
        return {
            "shape": list(value.shape),
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "norm": float(np.linalg.norm(value)),
        }

    @staticmethod
    def _activation(name: str, z, xp):
        key = (name or "linear").lower()
        if key == "relu":
            return xp.maximum(z, 0)
        if key == "leaky_relu":
            return xp.where(z > 0, z, 0.01 * z)
        if key == "sigmoid":
            return 1.0 / (1.0 + xp.exp(-xp.clip(z, -80, 80)))
        if key == "tanh":
            return xp.tanh(z)
        if key == "softmax":
            shifted = z - xp.max(z, axis=-1, keepdims=True)
            exps = xp.exp(shifted)
            return exps / xp.sum(exps, axis=-1, keepdims=True)
        if key == "linear":
            return z
        raise ValueError(f"Unsupported activation '{name}'.")

    @staticmethod
    def _activation_derivative(name: str, z, activated, xp):
        key = (name or "linear").lower()
        if key == "relu":
            return (z > 0).astype(xp.float32)
        if key == "leaky_relu":
            return xp.where(z > 0, 1.0, 0.01).astype(xp.float32)
        if key == "sigmoid":
            return activated * (1.0 - activated)
        if key == "tanh":
            return 1.0 - activated * activated
        if key == "linear":
            return xp.ones_like(z)
        if key == "softmax":
            raise ValueError("Softmax backward is supported only with BCE-compatible fused loss semantics, not generic elementwise derivatives.")
        raise ValueError(f"Unsupported activation '{name}'.")

    def build_graph(self, architecture: List[Dict]) -> Dict:
        if not isinstance(architecture, list) or len(architecture) < 2:
            raise ValueError("Architecture must contain an input layer and at least one dense/output layer.")
        if architecture[0].get("type") != "input":
            raise ValueError("The first layer must have type 'input'.")

        normalized = []
        for index, layer in enumerate(architecture):
            kind = str(layer.get("type", "dense")).lower()
            if kind not in self.SUPPORTED_LAYERS:
                raise ValueError(
                    f"Unsupported layer type '{kind}' at index {index}; this execution engine supports dense feed-forward layers only."
                )
            neurons = layer.get("neurons")
            if not isinstance(neurons, int) or isinstance(neurons, bool) or neurons < 1:
                raise ValueError(f"Layer {index} must define a positive integer neuron count.")
            activation = str(layer.get("activation", "linear" if index == 0 else "relu")).lower()
            if index > 0 and activation not in self.SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Unsupported activation '{activation}' at layer {index}.")
            if index == 0 and kind != "input":
                raise ValueError("The first layer must have type 'input'.")
            if index > 0 and kind == "input":
                raise ValueError("Only the first layer may have type 'input'.")
            normalized.append({"type": kind, "neurons": neurons, "activation": activation})

        self._backend = self._get_backend()
        xp = self._xp()
        rng = np.random.default_rng()
        weights, biases = [], []
        previous_dim = normalized[0]["neurons"]
        for layer in normalized[1:]:
            out_dim = layer["neurons"]
            activation = layer["activation"]
            scale = np.sqrt(2.0 / previous_dim) if activation in {"relu", "leaky_relu"} else np.sqrt(1.0 / previous_dim)
            w = rng.normal(0.0, scale, size=(previous_dim, out_dim)).astype(np.float32)
            b = np.zeros((out_dim,), dtype=np.float32)
            weights.append(xp.asarray(w))
            biases.append(xp.asarray(b))
            previous_dim = out_dim

        total_params = int(sum(int(w.size + b.size) for w, b in zip(weights, biases)))
        self._graph = {
            "weights": weights,
            "biases": biases,
            "layers": normalized,
            "total_params": total_params,
            "graph_id": str(uuid.uuid4()),
        }
        self._last_forward = None
        return {
            "graph_id": self._graph["graph_id"],
            "total_params": total_params,
            "layers": len(normalized),
            "device": self._backend,
        }

    def forward(self, input_data: List[float]) -> ForwardResult:
        if not self._graph:
            raise ValueError("No graph built. Call build_graph first.")
        start = time.perf_counter()
        xp = self._xp()
        x = xp.asarray(input_data, dtype=xp.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        expected = self._graph["layers"][0]["neurons"]
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(f"Input must have shape (batch, {expected}); received {tuple(x.shape)}.")

        activations = [x]
        pre_activations = []
        current = x
        layers_data = [LayerData(
            layer_index=0,
            layer_type="input",
            pre_activation=x,
            post_activation=x,
            activation_stats=self._compute_stats(x),
        )]
        for idx, (w, b, config) in enumerate(zip(
            self._graph["weights"], self._graph["biases"], self._graph["layers"][1:]
        )):
            z = current @ w + b
            a = self._activation(config["activation"], z, xp)
            pre_activations.append(z)
            activations.append(a)
            layers_data.append(LayerData(
                layer_index=idx + 1,
                layer_type=config["type"],
                pre_activation=z,
                post_activation=a,
                weights=w,
                biases=b,
                activation_stats=self._compute_stats(a),
                weight_stats=self._compute_stats(w),
            ))
            current = a

        self._last_forward = {
            "input": x.copy(),
            "activations": [value.copy() for value in activations],
            "pre_activations": [value.copy() for value in pre_activations],
        }
        output = self._to_numpy(current)
        return ForwardResult(
            layers=layers_data,
            final_output=output[0] if output.shape[0] == 1 else output,
            total_params=self._graph["total_params"],
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
            device_used=self._backend,
        )

    def backward(self, target: List[float], loss_function: str = "mse") -> BackwardResult:
        if not self._graph:
            raise ValueError("No graph built. Call build_graph first.")
        if self._last_forward is None:
            raise ValueError("Run forward before backward so the input and activations are available.")
        start = time.perf_counter()
        xp = self._xp()
        loss_name = (loss_function or "mse").lower()
        if loss_name not in self.SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss '{loss_function}'. Choose from {sorted(self.SUPPORTED_LOSSES)}.")

        cache = self._last_forward
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]
        y_hat = activations[-1]
        y = xp.asarray(target, dtype=xp.float32)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.shape != y_hat.shape:
            raise ValueError(f"Target shape {tuple(y.shape)} must match output shape {tuple(y_hat.shape)}.")

        output_activation = self._graph["layers"][-1]["activation"]
        if loss_name == "mse":
            loss = 0.5 * xp.mean((y_hat - y) ** 2)
            delta = ((y_hat - y) / y.size) * self._activation_derivative(
                output_activation, pre_activations[-1], y_hat, xp
            )
        elif output_activation == "sigmoid":
            clipped = xp.clip(y_hat, 1e-7, 1.0 - 1e-7)
            loss = -xp.mean(y * xp.log(clipped) + (1.0 - y) * xp.log(1.0 - clipped))
            delta = (y_hat - y) / y.size
        else:
            clipped = xp.clip(y_hat, 1e-7, 1.0 - 1e-7)
            loss = -xp.mean(y * xp.log(clipped) + (1.0 - y) * xp.log(1.0 - clipped))
            d_loss_da = -(y / clipped - (1.0 - y) / (1.0 - clipped)) / y.size
            delta = d_loss_da * self._activation_derivative(
                output_activation, pre_activations[-1], y_hat, xp
            )

        grad_w = [None] * len(self._graph["weights"])
        grad_b = [None] * len(self._graph["biases"])
        deltas = [None] * len(grad_w)
        for idx in reversed(range(len(grad_w))):
            deltas[idx] = delta
            grad_w[idx] = activations[idx].T @ delta
            grad_b[idx] = xp.sum(delta, axis=0)
            if idx > 0:
                previous_config = self._graph["layers"][idx]
                previous_z = pre_activations[idx - 1]
                previous_a = activations[idx]
                delta = (delta @ self._graph["weights"][idx].T) * self._activation_derivative(
                    previous_config["activation"], previous_z, previous_a, xp
                )

        layers_data = []
        total_squared_norm = 0.0
        for idx, config in enumerate(self._graph["layers"][1:]):
            gw, gb, d = grad_w[idx], grad_b[idx], deltas[idx]
            gw_np, gb_np = self._to_numpy(gw), self._to_numpy(gb)
            total_squared_norm += float(np.sum(gw_np ** 2) + np.sum(gb_np ** 2))
            layers_data.append(LayerData(
                layer_index=idx + 1,
                layer_type=config["type"],
                pre_activation=pre_activations[idx],
                post_activation=activations[idx + 1],
                gradient_weights=gw,
                gradient_bias=gb,
                deltas=d,
                weights=self._graph["weights"][idx],
                biases=self._graph["biases"][idx],
                gradient_stats=self._compute_stats(gw),
                weight_stats=self._compute_stats(self._graph["weights"][idx]),
            ))
        return BackwardResult(
            layers=layers_data,
            loss_value=float(self._to_numpy(loss)),
            total_gradient_norm=float(np.sqrt(total_squared_norm)),
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def train(
        self,
        dataset: Dict,
        epochs: int = 10,
        learning_rate: float = 0.01,
        loss_function: str = "mse",
    ) -> TrainingResult:
        if not self._graph:
            raise ValueError("No graph built. Call build_graph first.")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(dataset, dict) or "x_train" not in dataset or "y_train" not in dataset:
            raise ValueError("dataset must contain x_train and y_train arrays.")
        x_train = np.asarray(dataset["x_train"], dtype=np.float32)
        y_train = np.asarray(dataset["y_train"], dtype=np.float32)
        if x_train.ndim == 1:
            x_train = x_train.reshape(1, -1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(1, -1)
        if x_train.ndim != 2 or y_train.ndim != 2 or x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must be 2D arrays with the same sample count.")
        if x_train.shape[0] == 0:
            raise ValueError("Training dataset must contain at least one sample.")

        xp = self._xp()
        x_dev, y_dev = xp.asarray(x_train), xp.asarray(y_train)
        start = time.perf_counter()
        loss_history: List[float] = []
        metrics_history: List[Dict[str, float]] = []
        for epoch in range(epochs):
            forward = self.forward(x_dev)
            backward = self.backward(y_dev, loss_function=loss_function)
            train_loss = backward.loss_value
            # Recompute gradients from the cached forward pass before updating
            # parameters. This keeps the update tied to the scalar loss contract.
            gradients = self._compute_gradients(y_dev, loss_function)
            for idx in range(len(self._graph["weights"])):
                self._graph["weights"][idx] = self._graph["weights"][idx] - learning_rate * gradients[0][idx]
                self._graph["biases"][idx] = self._graph["biases"][idx] - learning_rate * gradients[1][idx]
            loss_history.append(float(train_loss))
            post_update = self.forward(x_dev)
            predictions = xp.asarray(post_update.final_output, dtype=xp.float32)
            if predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)
            if predictions.shape[1] == 1:
                predicted_labels = (predictions[:, 0] >= 0.5).astype(xp.int32)
                true_labels = (y_dev[:, 0] >= 0.5).astype(xp.int32)
            else:
                predicted_labels = xp.argmax(predictions, axis=1)
                true_labels = xp.argmax(y_dev, axis=1)
            accuracy = float(self._to_numpy(xp.mean(predicted_labels == true_labels)))
            metrics_history.append({
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "train_accuracy": accuracy,
            })
        final_metrics = metrics_history[-1] if metrics_history else {}
        return TrainingResult(
            loss_history=loss_history,
            metrics_history=metrics_history,
            final_metrics=final_metrics,
            total_epochs=epochs,
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _compute_gradients(self, target, loss_function="mse"):
        """Return current gradients without mutating graph parameters."""
        if self._last_forward is None:
            raise ValueError("Run forward before requesting gradients.")
        xp = self._xp()
        cache = self._last_forward
        activations, pre_activations = cache["activations"], cache["pre_activations"]
        y_hat = activations[-1]
        y = xp.asarray(target, dtype=xp.float32)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.shape != y_hat.shape:
            raise ValueError(f"Target shape {tuple(y.shape)} must match output shape {tuple(y_hat.shape)}.")
        loss_name = (loss_function or "mse").lower()
        output_activation = self._graph["layers"][-1]["activation"]
        if loss_name == "mse":
            delta = ((y_hat - y) / y.size) * self._activation_derivative(output_activation, pre_activations[-1], y_hat, xp)
        elif loss_name == "bce" and output_activation == "sigmoid":
            delta = (y_hat - y) / y.size
        elif loss_name == "bce":
            clipped = xp.clip(y_hat, 1e-7, 1.0 - 1e-7)
            d_loss_da = -(y / clipped - (1.0 - y) / (1.0 - clipped)) / y.size
            delta = d_loss_da * self._activation_derivative(output_activation, pre_activations[-1], y_hat, xp)
        else:
            raise ValueError(f"Unsupported loss '{loss_function}'.")
        grad_w = [None] * len(self._graph["weights"])
        grad_b = [None] * len(self._graph["biases"])
        for idx in reversed(range(len(grad_w))):
            grad_w[idx] = activations[idx].T @ delta
            grad_b[idx] = xp.sum(delta, axis=0)
            if idx > 0:
                delta = (delta @ self._graph["weights"][idx].T) * self._activation_derivative(
                    self._graph["layers"][idx]["activation"],
                    pre_activations[idx - 1],
                    activations[idx],
                    xp,
                )
        return grad_w, grad_b

    def get_device_info(self) -> Dict:
        return {
            "backend": self._backend,
            "cupy_available": CUPY_AVAILABLE,
            "device_preference": self._device_preference,
        }

    def set_device(self, device: str) -> None:
        self._device_preference = device
        self._backend = self._get_backend()
        self._last_forward = None

    def get_layer_data_json(self, result: ForwardResult) -> List[Dict]:
        layers_json = []
        for layer in result.layers:
            layer_dict = {"layer_index": layer.layer_index, "type": layer.layer_type}
            if layer.post_activation is not None:
                arr = self._to_numpy(layer.post_activation)
                layer_dict["activation"] = {
                    "shape": list(arr.shape),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "sample": arr.flatten()[:10].tolist(),
                }
            if layer.weights is not None:
                arr = self._to_numpy(layer.weights)
                layer_dict["weights"] = {
                    "shape": list(arr.shape),
                    "stats": {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    },
                }
            if layer.gradient_weights is not None:
                arr = self._to_numpy(layer.gradient_weights)
                layer_dict["gradient"] = {
                    "shape": list(arr.shape),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "norm": float(np.linalg.norm(arr)),
                }
            layers_json.append(layer_dict)
        return layers_json


execution_engine = ExecutionEngine()


def get_engine() -> ExecutionEngine:
    return execution_engine
