from __future__ import annotations

from typing import Dict, List

import numpy as np

from .activations import get_activation
from .graph_engine import NetworkGraph
from .loss_functions import compute_loss


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def backward_full(
    graph: NetworkGraph,
    input_vec: List[float],
    target: List[float],
    loss_fn: str,
    l2_lambda: float = 0.0,
) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32).reshape(-1)

    graph.forward(x)
    y_hat = graph.activations[-1].reshape(-1)
    loss_value, loss_eq = compute_loss(y, y_hat, loss_fn)

    dense_only = all(layer.layer_type in {"input", "dense", "output"} for layer in graph.layers)
    if dense_only:
        deltas: List[np.ndarray] = []
        grads_w: List[np.ndarray] = [np.zeros_like(w) for w in graph.weights]
        grads_b: List[np.ndarray] = [np.zeros_like(b) for b in graph.biases]
        steps: List[Dict] = []

        num_layers = len(graph.weights)
        step_index = 0

        for layer_idx in reversed(range(num_layers)):
            z = graph.pre_activations[layer_idx]
            a_prev = graph.activations[layer_idx]
            act_name = (graph.layers[layer_idx + 1].activation or "linear").lower()
            activation = get_activation(act_name)

            if layer_idx == num_layers - 1:
                if loss_fn == "mse":
                    delta = ((y_hat - y) / max(1, y.size)) * activation.derivative(z)
                else:
                    delta = y_hat - y
            else:
                w_next = graph.weights[layer_idx + 1]
                delta = (w_next.T @ deltas[-1]) * activation.derivative(z)

            deltas.append(delta)

            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx,
                    "operation": "compute_delta",
                    "description": f"Layer {layer_idx + 1}: Compute delta",
                    "equation_text": "delta = (W.T�delta_next) ? f'(z)" if layer_idx < num_layers - 1 else "delta = a - y",
                    "equation_detailed": [f"delta{j+1} = {_fmt(delta[j])}" for j in range(min(4, len(delta)))],
                    "output_values": delta.tolist(),
                }
            )
            step_index += 1

            dW = np.outer(delta, a_prev)
            if l2_lambda > 0.0:
                dW = dW + l2_lambda * graph.weights[layer_idx]
            db = delta.copy()
            grads_w[layer_idx] = dW
            grads_b[layer_idx] = db

            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx,
                    "operation": "compute_dW",
                    "description": f"Layer {layer_idx + 1}: Weight Gradients",
                    "equation_text": "?L/?W = delta � a_prev^T",
                    "equation_detailed": [
                        f"?w{r+1}{c+1} = {_fmt(dW[r, c])}"
                        for r in range(min(2, dW.shape[0]))
                        for c in range(min(2, dW.shape[1]))
                    ],
                    "gradient_values": dW.tolist(),
                }
            )
            step_index += 1

            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx,
                    "operation": "compute_db",
                    "description": f"Layer {layer_idx + 1}: Bias Gradients",
                    "equation_text": "?L/?b = delta",
                    "equation_detailed": [f"?b{j+1} = {_fmt(db[j])}" for j in range(min(4, len(db)))],
                    "output_values": db.tolist(),
                }
            )
            step_index += 1

        deltas = list(reversed(deltas))
        
        

        per_layer = []
        all_grads = np.concatenate([g.flatten() for g in grads_w]) if grads_w else np.array([0.0])
        for idx, (dw, db, delta) in enumerate(zip(grads_w, grads_b, deltas)):
            per_layer.append(
                {
                    "layer": idx,
                    "dW_norm": float(np.linalg.norm(dw)),
                    "db_norm": float(np.linalg.norm(db)),
                    "dW_mean": float(np.mean(dw)),
                    "dW_std": float(np.std(dw)),
                    "dW_min": float(np.min(dw)),
                    "dW_max": float(np.max(dw)),
                    "delta_norm": float(np.linalg.norm(delta)),
                }
            )

        if graph.weights:
            d_input = graph.weights[0].T @ deltas[0]
        else:
            d_input = np.zeros_like(x)

        return {
            "input_gradient": np.asarray(d_input).reshape(-1).tolist(),
            "loss_value": loss_value,
            "loss_equation": loss_eq,
            "steps": steps,
            "total_steps": len(steps),
            "gradients_W": [g.tolist() for g in grads_w],
            "gradients_b": [g.tolist() for g in grads_b],
            "deltas": [d.tolist() for d in deltas],
            "gradient_summary": {
                "per_layer": per_layer,
                "total_gradient_norm": float(np.linalg.norm(all_grads)),
                "max_gradient": float(np.max(all_grads)),
                "min_gradient": float(np.min(all_grads)),
            },
        }

    steps: List[Dict] = []
    step_index = 0
    grads_by_layer: Dict[int, Dict[str, np.ndarray]] = {}

    if loss_fn == "mse":
        # DenseLayer.backward applies the output activation derivative itself.
        # Seed it with dL/da, not dL/dz, to avoid applying phi'(z) twice.
        d_out = (y_hat - y) / max(1, y.size)
    else:
        d_out = y_hat - y

    for layer_idx in reversed(range(1, len(graph.layer_instances))):
        layer = graph.layer_instances[layer_idx]
        d_out = layer.backward(d_out)
        steps.append(
            {
                "step_index": step_index,
                "layer_index": layer_idx - 1,
                "operation": "compute_delta",
                "description": f"Layer {layer_idx}: Backward",
                "output_values": np.asarray(d_out).reshape(-1).tolist(),
            }
        )
        step_index += 1

        if getattr(layer, "has_params", False):
            dw, db = layer.grads()
            if dw is not None and l2_lambda > 0.0:
                dw = dw + l2_lambda * layer.params()[0]
            grads_by_layer[layer_idx] = {"dW": dw, "db": db}
            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx - 1,
                    "operation": "compute_dW",
                    "description": f"Layer {layer_idx}: Weight Gradients",
                    "gradient_values": np.asarray(dw).reshape(-1).tolist() if dw is not None else [],
                }
            )
            step_index += 1
            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx - 1,
                    "operation": "compute_db",
                    "description": f"Layer {layer_idx}: Bias Gradients",
                    "output_values": np.asarray(db).reshape(-1).tolist() if db is not None else [],
                }
            )
            step_index += 1

    grads_w: List[np.ndarray] = []
    grads_b: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    for layer_idx in graph.param_layer_indices:
        grads = grads_by_layer.get(layer_idx)
        if not grads:
            continue
        grads_w.append(grads["dW"])
        grads_b.append(grads["db"])

    all_grads = np.concatenate([g.flatten() for g in grads_w]) if grads_w else np.array([0.0])
    per_layer = []
    for idx, (dw, db) in enumerate(zip(grads_w, grads_b)):
        per_layer.append(
            {
                "layer": idx,
                "dW_norm": float(np.linalg.norm(dw)),
                "db_norm": float(np.linalg.norm(db)),
                "dW_mean": float(np.mean(dw)),
                "dW_std": float(np.std(dw)),
                "dW_min": float(np.min(dw)),
                "dW_max": float(np.max(dw)),
                "delta_norm": float(np.linalg.norm(dw)),
            }
        )

    d_input = np.asarray(d_out).reshape(-1)

    return {
        "input_gradient": d_input.tolist(),
        "loss_value": loss_value,
        "loss_equation": loss_eq,
        "steps": steps,
        "total_steps": len(steps),
        "gradients_W": [g.tolist() for g in grads_w],
        "gradients_b": [g.tolist() for g in grads_b],
        "deltas": [d.tolist() for d in deltas],
        "gradient_summary": {
            "per_layer": per_layer,
            "total_gradient_norm": float(np.linalg.norm(all_grads)),
            "max_gradient": float(np.max(all_grads)) if all_grads.size else 0.0,
            "min_gradient": float(np.min(all_grads)) if all_grads.size else 0.0,
        },
    }


def backward_step(graph: NetworkGraph, input_vec: List[float], target: List[float], loss_fn: str, step_index: int) -> Dict:
    result = backward_full(graph, input_vec, target, loss_fn)
    steps = result["steps"]
    if step_index < 0 or step_index >= len(steps):
        raise IndexError("step_index out of range")

    completed_layers = sorted({s["layer_index"] for s in steps[: step_index + 1] if s["operation"] == "compute_db"})
    active_layer = steps[step_index]["layer_index"]
    partial = {str(i): None for i in range(len(result["gradients_W"]))}
    for i in completed_layers:
        if i < len(result["gradients_W"]):
            partial[str(i)] = {"dW": result["gradients_W"][i], "db": result["gradients_b"][i]}

    return {
        "input_gradient": result.get("input_gradient", []),
        "step": steps[step_index],
        "completed_layers": completed_layers,
        "active_layer": active_layer,
        "partial_gradients": partial,
        "loss_value": result["loss_value"],
        "loss_equation": result["loss_equation"],
    }


