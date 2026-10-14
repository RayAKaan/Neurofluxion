from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

import numpy as np

from .activations import get_activation
from .dropout_engine import apply_dropout
from .loss_functions import compute_loss
from .lr_scheduler import get_lr
from .optimizer_engine import OptimizerState, apply_update
from .snapshot_manager import Snapshot, snapshot_manager


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "adam"
    loss_function: str = "bce"
    l2_lambda: float = 0.0
    dropout_rate: float = 0.0
    lr_scheduler: str | None = None
    lr_decay_rate: float | None = None
    lr_step_size: int | None = None
    shuffle: bool = True
    snapshot_interval: int = 5


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    learning_rate: float
    gradient_norms: List[float]
    weight_norms: List[float]
    weight_deltas: List[float]
    dead_neurons: List[int]
    batch_losses: List[float]
    epoch_duration_ms: float


class TrainingEngine:
    def __init__(self, graph, config: TrainingConfig):
        self.graph = graph
        self.config = config
        self.optimizer_state: OptimizerState | None = None
        self.epoch = 0
        self.history: List[TrainingMetrics] = []
        self.is_running = False
        self.is_paused = False
        self.last_gradients: Dict[str, List[np.ndarray]] = {"dW": [], "db": []}
        self.weight_history: List[Dict] = []

    def _uses_layer_instances(self) -> bool:
        return bool(self.graph.layer_instances)

    def _forward(self, x: np.ndarray, training: bool) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if not self._uses_layer_instances():
            activations = [x]
            pre_acts = []
            masks = []
            for idx in range(len(self.graph.weights)):
                w = self.graph.weights[idx]
                b = self.graph.biases[idx]
                z = w @ activations[-1] + b
                pre_acts.append(z)
                act_name = (self.graph.layers[idx + 1].activation or "linear").lower()
                a = get_activation(act_name).forward(z)
                if training and self.config.dropout_rate > 0.0 and idx < len(self.graph.weights) - 1:
                    a, mask = apply_dropout(a, self.config.dropout_rate)
                else:
                    mask = np.ones_like(a)
                masks.append(mask)
                activations.append(a)
            return activations, pre_acts, masks

        activations = [x]
        pre_acts = []
        masks = []
        current = x
        for idx in range(1, len(self.graph.layer_instances)):
            layer = self.graph.layer_instances[idx]
            current = layer.forward(current)
            if hasattr(layer, "Z") and getattr(layer, "Z") is not None:
                pre_acts.append(layer.Z)
            if training and self.config.dropout_rate > 0.0 and idx < len(self.graph.layer_instances) - 1:
                current, mask = apply_dropout(current, self.config.dropout_rate)
            else:
                mask = np.ones_like(current)
            masks.append(mask)
            activations.append(current)
        return activations, pre_acts, masks

    def _backward(
        self,
        activations: List[np.ndarray],
        pre_acts: List[np.ndarray],
        masks: List[np.ndarray],
        y: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not self._uses_layer_instances():
            num_layers = len(self.graph.weights)
            grads_w = [np.zeros_like(w) for w in self.graph.weights]
            grads_b = [np.zeros_like(b) for b in self.graph.biases]
            deltas: List[np.ndarray] = []

            y_hat = activations[-1]
            for idx in reversed(range(num_layers)):
                z = pre_acts[idx]
                act_name = (self.graph.layers[idx + 1].activation or "linear").lower()
                activation = get_activation(act_name)
                if idx == num_layers - 1:
                    if (self.config.loss_function or "bce").lower() == "mse":
                        delta = ((y_hat - y) / max(1, y.size)) * activation.derivative(z)
                    elif (self.config.loss_function or "bce").lower() == "bce" and act_name == "sigmoid":
                        # Mean BCE with sigmoid output uses the fused dL/dz = (y_hat-y)/N.
                        delta = (y_hat - y) / max(1, y.size)
                    else:
                        delta = y_hat - y
                else:
                    w_next = self.graph.weights[idx + 1]
                    delta = (w_next.T @ deltas[-1]) * activation.derivative(z)
                    if self.config.dropout_rate > 0.0:
                        keep_prob = 1.0 - self.config.dropout_rate
                        delta = delta * masks[idx] / keep_prob
                deltas.append(delta)
                grads_w[idx] = np.outer(delta, activations[idx])
                if self.config.l2_lambda > 0.0:
                    grads_w[idx] += self.config.l2_lambda * self.graph.weights[idx]
                grads_b[idx] = delta.copy()
            return grads_w, grads_b

        grads_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
        y_hat = activations[-1].reshape(-1)
        last_layer = self.graph.layer_instances[-1]
        if hasattr(last_layer, "activation_name"):
            act = get_activation(last_layer.activation_name)
            if (self.config.loss_function or "bce").lower() == "mse" and hasattr(last_layer, "Z") and last_layer.Z is not None:
                d_out = (y_hat - y) / max(1, y.size)
            elif (self.config.loss_function or "bce").lower() == "bce" and last_layer.activation_name == "sigmoid" and getattr(last_layer, "Z", None) is not None:
                # DenseLayer.backward multiplies sigmoid'; compensate so the fused BCE/sigmoid dL/dz is retained.
                derivative = act.derivative(last_layer.Z)
                d_out = ((y_hat - y) / max(1, y.size)) / np.maximum(derivative, np.finfo(np.float32).eps)
            else:
                d_out = y_hat - y
        else:
            d_out = y_hat - y

        for idx in reversed(range(1, len(self.graph.layer_instances))):
            if self.config.dropout_rate > 0.0 and idx < len(self.graph.layer_instances) - 1:
                keep_prob = 1.0 - self.config.dropout_rate
                d_out = d_out * masks[idx - 1] / keep_prob
            layer = self.graph.layer_instances[idx]
            d_out = layer.backward(d_out)
            if getattr(layer, "has_params", False):
                dw, db = layer.grads()
                if dw is not None and self.config.l2_lambda > 0.0:
                    dw = dw + self.config.l2_lambda * layer.params()[0]
                grads_by_layer[idx] = {"dW": dw, "db": db}

        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []
        for layer_idx in self.graph.param_layer_indices:
            grads = grads_by_layer.get(layer_idx)
            if not grads:
                continue
            grads_w.append(grads["dW"])
            grads_b.append(grads["db"])
        return grads_w, grads_b

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.shape[0] == 1:
            pred = (y_pred[0] > 0.5).astype(np.float32)
            return float(pred == y_true[0])
        pred = int(np.argmax(y_pred))
        true = int(np.argmax(y_true))
        return float(pred == true)

    def train_batch(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float, float]:
        batch_loss = 0.0
        batch_acc = 0.0
        grads_w_sum = [np.zeros_like(w) for w in self.graph.weights]
        grads_b_sum = [np.zeros_like(b) for b in self.graph.biases]

        for x, y in batch:
            activations, pre_acts, masks = self._forward(x, training=True)
            y_hat = activations[-1].reshape(-1)
            loss, _ = compute_loss(y, y_hat, self.config.loss_function)
            grads_w, grads_b = self._backward(activations, pre_acts, masks, y)
            grads_w_sum = [a + b for a, b in zip(grads_w_sum, grads_w)]
            grads_b_sum = [a + b for a, b in zip(grads_b_sum, grads_b)]
            batch_loss += loss
            batch_acc += self._accuracy(y, y_hat)

        batch_size = max(1, len(batch))
        grads_w_avg = [g / batch_size for g in grads_w_sum]
        grads_b_avg = [g / batch_size for g in grads_b_sum]

        lr = get_lr(self.config.learning_rate, self.epoch, self.config.lr_scheduler, self.config.lr_decay_rate, self.config.lr_step_size)
        old_weights = [w.copy() for w in self.graph.weights]
        new_w, new_b, self.optimizer_state = apply_update(
            self.graph.weights, self.graph.biases, grads_w_avg, grads_b_avg, lr, self.config.optimizer, self.optimizer_state
        )
        self.graph.weights = new_w
        self.graph.biases = new_b
        if self._uses_layer_instances():
            for layer_idx, param_idx in self.graph.param_index_by_layer.items():
                layer = self.graph.layer_instances[layer_idx]
                layer.set_params(self.graph.weights[param_idx], self.graph.biases[param_idx])
        weight_deltas = [float(np.linalg.norm(n - o)) for n, o in zip(new_w, old_weights)]

        self.last_gradients = {"dW": grads_w_avg, "db": grads_b_avg}
        grad_norm = float(np.linalg.norm(np.concatenate([g.flatten() for g in grads_w_avg]))) if grads_w_avg else 0.0

        self.weight_history.append(
            {
                "epoch": self.epoch,
                "weight_norms": [float(np.linalg.norm(w)) for w in self.graph.weights],
                "weight_deltas": weight_deltas,
            }
        )

        return batch_loss / batch_size, batch_acc / batch_size, grad_norm

    def compute_test_metrics(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
        losses = []
        accs = []
        for x, y in data:
            activations, _, _ = self._forward(x, training=False)
            y_hat = activations[-1].reshape(-1)
            loss, _ = compute_loss(y, y_hat, self.config.loss_function)
            losses.append(loss)
            accs.append(self._accuracy(y, y_hat))
        if not losses:
            return 0.0, 0.0
        return float(np.mean(losses)), float(np.mean(accs))

    def train_epoch(
        self,
        train_data: List[Tuple[np.ndarray, np.ndarray]],
        test_data: List[Tuple[np.ndarray, np.ndarray]],
        graph_id: str,
        batch_callback=None,
    ) -> TrainingMetrics:
        start = time.time()
        if self.config.shuffle:
            np.random.shuffle(train_data)
        batch_size = self.config.batch_size if self.config.batch_size > 0 else len(train_data)
        batches = [train_data[i : i + batch_size] for i in range(0, len(train_data), batch_size)]

        batch_losses: List[float] = []
        grad_norms = []
        for bi, batch in enumerate(batches):
            loss, acc, grad_norm = self.train_batch(batch)
            batch_losses.append(loss)
            grad_norms.append(grad_norm)
            if batch_callback:
                batch_callback(bi, len(batches), loss, acc, grad_norm)

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        train_acc = float(np.mean([self._accuracy(y, self._forward(x, training=False)[0][-1].reshape(-1)) for x, y in train_data])) if train_data else 0.0
        test_loss, test_acc = self.compute_test_metrics(test_data)

        lr = get_lr(self.config.learning_rate, self.epoch, self.config.lr_scheduler, self.config.lr_decay_rate, self.config.lr_step_size)
        weight_norms = [float(np.linalg.norm(w)) for w in self.graph.weights]
        weight_deltas = self.weight_history[-1]["weight_deltas"] if self.weight_history else [0.0] * len(weight_norms)
        dead_neurons = [int(np.sum(self._forward(x, training=False)[0][i + 1] == 0.0)) for i in range(len(self.graph.weights)) for x, _ in train_data[:1]]
        dead_neurons = dead_neurons[: len(self.graph.weights)]

        metrics = TrainingMetrics(
            epoch=self.epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            learning_rate=lr,
            gradient_norms=[float(np.mean(grad_norms))] * len(weight_norms),
            weight_norms=weight_norms,
            weight_deltas=weight_deltas,
            dead_neurons=dead_neurons,
            batch_losses=batch_losses,
            epoch_duration_ms=(time.time() - start) * 1000.0,
        )
        self.history.append(metrics)

        if self.config.snapshot_interval and (self.epoch % self.config.snapshot_interval == 0 or self.epoch == 0):
            snapshot_manager.add_snapshot(
                graph_id,
                Snapshot(
                    epoch=self.epoch,
                    weights=[w.copy() for w in self.graph.weights],
                    biases=[b.copy() for b in self.graph.biases],
                    metrics=metrics.__dict__,
                ),
            )

        self.epoch += 1
        return metrics


class TrainingSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, TrainingEngine] = {}

    def get_or_create(self, graph_id: str, graph, config: TrainingConfig | None = None) -> TrainingEngine:
        if graph_id not in self._sessions:
            cfg = config if config else TrainingConfig()
            self._sessions[graph_id] = TrainingEngine(graph, cfg)
        return self._sessions[graph_id]

    def get(self, graph_id: str) -> TrainingEngine:
        if graph_id not in self._sessions:
            raise KeyError("Training session not found")
        return self._sessions[graph_id]

    def reset(self, graph_id: str, graph, config: TrainingConfig) -> TrainingEngine:
        self._sessions[graph_id] = TrainingEngine(graph, config)
        return self._sessions[graph_id]


training_sessions = TrainingSessionManager()
