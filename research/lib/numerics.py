"""Numerical comparison helpers: error metrics, finite differences, oracles.

Design rules encoded here:

* The finite-difference oracle differentiates the *implementation's own*
  scalar objective, because that is the function the simulator claims to
  minimise. An independent reference objective is compared separately so a
  forward-pass error cannot be masked by a matching loss definition.
* Parameter synchronisation is always *verified*, never assumed. The graph
  packs parameters into flat arrays while each layer owns its own copy; a
  silent divergence between the two is itself a finding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------
def as_float_array(value: object) -> Array:
    return np.asarray(value, dtype=np.float64)


def max_abs_error(actual: object, expected: object) -> Optional[float]:
    a = as_float_array(actual)
    b = as_float_array(expected)
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def relative_l2(actual: object, expected: object, eps: float = 1e-12) -> Optional[float]:
    a = as_float_array(actual).ravel()
    b = as_float_array(expected).ravel()
    if a.shape != b.shape or a.size == 0:
        return None
    denom = float(np.linalg.norm(a) + np.linalg.norm(b) + eps)
    return float(np.linalg.norm(a - b) / denom)


def cosine_similarity(actual: object, expected: object) -> Optional[float]:
    a = as_float_array(actual).ravel()
    b = as_float_array(expected).ravel()
    if a.size == 0 or a.shape != b.shape:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def passes_allclose(
    actual: object,
    expected: object,
    atol: float,
    rtol: float,
) -> Tuple[bool, Optional[float]]:
    a = as_float_array(actual)
    b = as_float_array(expected)
    if a.shape != b.shape:
        return False, None
    if a.size == 0:
        return True, 0.0
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    return ok, float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------------
# parameter synchronisation
# ---------------------------------------------------------------------------
@dataclass
class ParamSlot:
    """One differentiable scalar addressable inside the packed graph arrays."""

    param_index: int
    kind: str  # "weights" | "biases"
    layer_instance_index: int
    layer_type: str
    shape: Tuple[int, ...]
    flat_offset: int

    @property
    def name(self) -> str:
        return f"{self.kind}[{self.param_index}]"


@dataclass
class SyncReport:
    """Result of writing packed graph parameters back into layer instances."""

    attempted: int = 0
    verified: int = 0
    desynchronised: List[Dict[str, object]] = field(default_factory=list)
    details: List[Dict[str, object]] = field(default_factory=list)

    @property
    def fully_synchronised(self) -> bool:
        return not self.desynchronised

    def to_dict(self) -> Dict[str, object]:
        return {
            "attempted": self.attempted,
            "verified": self.verified,
            "fully_synchronised": self.fully_synchronised,
            "desynchronised_count": len(self.desynchronised),
            "desynchronised": self.desynchronised,
            "details": self.details,
        }


def param_slots(graph) -> List[ParamSlot]:
    """Enumerate packed graph parameters with owning layer metadata."""
    slots: List[ParamSlot] = []
    instances = list(getattr(graph, "layer_instances", []) or [])
    param_layer_indices = list(getattr(graph, "param_layer_indices", []) or [])
    param_index_by_layer = dict(getattr(graph, "param_index_by_layer", {}) or {})
    offset = 0
    for kind, arrays in (("weights", graph.weights), ("biases", graph.biases)):
        for param_index, array in enumerate(arrays):
            layer_instance_index = -1
            layer_type = "unknown"
            for candidate_index, candidate_param in param_index_by_layer.items():
                if candidate_param == param_index and kind == "weights":
                    layer_instance_index = candidate_index
                    break
            if layer_instance_index >= 0 and layer_instance_index < len(instances):
                layer_type = getattr(
                    instances[layer_instance_index], "layer_type", "unknown"
                )
            elif param_index < len(param_layer_indices):
                layer_type = getattr(
                    instances[param_layer_indices[param_index]],
                    "layer_type",
                    "unknown",
                )
            slots.append(
                ParamSlot(
                    param_index=param_index,
                    kind=kind,
                    layer_instance_index=layer_instance_index,
                    layer_type=layer_type,
                    shape=tuple(int(v) for v in np.shape(array)),
                    flat_offset=offset,
                )
            )
            offset += int(np.size(array))
    return slots


def sync_params_to_layers(graph) -> SyncReport:
    """Push packed graph arrays into layer instances and verify the write.

    ``TrainingEngine.train_batch`` performs the same push. This helper exists so
    experiments can detect when that push silently fails to reach the layer that
    owns the parameter, instead of measuring a computation that no longer depends
    on the array being perturbed.
    """
    report = SyncReport()
    instances = list(getattr(graph, "layer_instances", []) or [])
    for layer_index, param_index in dict(
        getattr(graph, "param_index_by_layer", {}) or {}
    ).items():
        if layer_index >= len(instances):
            continue
        layer = instances[layer_index]
        w = graph.weights[param_index]
        b = graph.biases[param_index]
        before_w = np.asarray(layer.params()[0], dtype=np.float64).copy()
        before_b = np.asarray(layer.params()[1], dtype=np.float64).copy()
        report.attempted += 1
        try:
            layer.set_params(w.copy(), b.copy())
        except Exception as exc:  # pragma: no cover - defensive
            report.desynchronised.append(
                {
                    "layer_instance_index": layer_index,
                    "param_index": param_index,
                    "layer_type": getattr(layer, "layer_type", "unknown"),
                    "reason": "set_params raised",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        after_w = np.asarray(layer.params()[0], dtype=np.float64)
        after_b = np.asarray(layer.params()[1], dtype=np.float64)
        matched_w = before_w.shape == after_w.shape and bool(
            np.allclose(after_w, as_float_array(w), atol=0.0, rtol=0.0)
        )
        matched_b = before_b.shape == after_b.shape and bool(
            np.allclose(after_b, as_float_array(b), atol=0.0, rtol=0.0)
        )
        if matched_w and matched_b:
            report.verified += 1
        else:
            report.desynchronised.append(
                {
                    "layer_instance_index": layer_index,
                    "param_index": param_index,
                    "layer_type": getattr(layer, "layer_type", "unknown"),
                    "reason": "layer params unchanged after set_params",
                    "packed_shape": [int(v) for v in np.shape(w)],
                    "layer_shape_after": [int(v) for v in after_w.shape],
                    "max_abs_difference_after_write": float(
                        np.max(np.abs(after_w - as_float_array(w)))
                    )
                    if after_w.shape == np.shape(w)
                    else None,
                    "layer_params_changed_at_all": bool(
                        before_w.shape == after_w.shape
                        and not np.array_equal(before_w, after_w)
                    ),
                }
            )
        report.details.append(
            {
                "layer_instance_index": layer_index,
                "param_index": param_index,
                "layer_type": getattr(layer, "layer_type", "unknown"),
                "verified": bool(matched_w and matched_b),
            }
        )
    return report


def graph_forward(graph, x: Array) -> Array:
    return np.asarray(graph.forward(np.asarray(x, dtype=np.float32)), dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# finite differences
# ---------------------------------------------------------------------------
@dataclass
class FiniteDifferenceResult:
    epsilon: float
    numeric: Array
    per_element_abs_error: Array
    per_element_rel_error: Array
    numeric_norm: float
    max_abs_error: float
    relative_l2_error: float
    cosine_similarity: float
    passed: bool
    nonfinite_numeric_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "numeric_norm": self.numeric_norm,
            "max_abs_error": self.max_abs_error,
            "relative_l2_error": self.relative_l2_error,
            "cosine_similarity": self.cosine_similarity,
            "passed": self.passed,
            "nonfinite_numeric_count": self.nonfinite_numeric_count,
        }


def central_difference(
    objective: Callable[[], float],
    array: Array,
    epsilon: float,
    setter: Optional[Callable[[Array], None]] = None,
) -> Array:
    """Central difference of ``objective`` with respect to every element.

    ``setter`` receives the whole working array after every perturbation. It is
    required whenever the objective reads the parameter through an indirection
    such as a packed graph array plus a layer-instance copy; without it the
    numeric derivative would silently evaluate an unchanged objective.
    """
    work = np.array(array, dtype=np.float64, copy=True)
    grad = np.zeros_like(work)

    def commit() -> None:
        if setter is not None:
            setter(work)

    for index in np.ndindex(work.shape):
        original = float(work[index])
        try:
            work[index] = original + epsilon
            commit()
            plus = objective()
            work[index] = original - epsilon
            commit()
            minus = objective()
        finally:
            work[index] = original
            commit()
        grad[index] = (plus - minus) / (2.0 * epsilon)
    return grad


def compare_gradients(
    analytic: Sequence[float],
    numeric: Array,
    epsilon: float,
    atol: float,
    rtol: float,
) -> FiniteDifferenceResult:
    a = as_float_array(analytic).ravel()
    n = as_float_array(numeric).ravel()
    if a.shape != n.shape:
        raise ValueError(
            f"analytic/numeric shape mismatch: {a.shape} vs {n.shape}"
        )
    diff = np.abs(a - n)
    denom = np.maximum(np.abs(a), np.abs(n))
    rel = np.where(denom > 0, diff / np.maximum(denom, 1e-12), diff)
    passed = bool(np.allclose(a, n, atol=atol, rtol=rtol))
    return FiniteDifferenceResult(
        epsilon=epsilon,
        numeric=n,
        per_element_abs_error=diff,
        per_element_rel_error=rel,
        numeric_norm=float(np.linalg.norm(n)),
        max_abs_error=float(np.max(diff)) if diff.size else 0.0,
        relative_l2_error=float(np.linalg.norm(a - n) / (np.linalg.norm(a) + 1e-12))
        if a.size
        else 0.0,
        cosine_similarity=cosine_similarity(a, n),
        passed=passed,
        nonfinite_numeric_count=int(np.count_nonzero(~np.isfinite(n))),
    )


@dataclass
class ResolutionProbe:
    """Is a finite-difference comparison resolvable in the objective's dtype?

    When the analytic gradient is small relative to the objective's own
    floating-point resolution, the numeric derivative is dominated by round-off
    and a mismatch is evidence about the estimator, not about the implementation.
    """

    expected_signal: float
    observed_signal: float
    dtype_resolution: float
    signal_to_resolution: float
    resolvable: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "expected_signal": self.expected_signal,
            "observed_signal": self.observed_signal,
            "dtype_resolution": self.dtype_resolution,
            "signal_to_resolution": self.signal_to_resolution,
            "resolvable": self.resolvable,
            "note": (
                "expected_signal is 2*eps*|<analytic, direction>|. A ratio below "
                "10 means the central difference cannot resolve the analytic "
                "gradient in this dtype, so a mismatch is inconclusive."
            ),
        }


def resolution_probe(
    objective: Callable[[], float],
    setter: Optional[Callable[[Array], None]],
    base: Array,
    analytic: Sequence[float],
    epsilon: float,
    seed: int = 0,
    dtype: str = "float32",
    resolvability_ratio: float = 10.0,
) -> ResolutionProbe:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=np.shape(base))
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        direction = np.ones_like(direction)
        norm = float(np.linalg.norm(direction))
    direction = direction / norm

    expected = 2.0 * epsilon * abs(float(np.dot(np.asarray(analytic).ravel(), direction.ravel())))
    if setter is not None:
        plus_array = np.asarray(base, dtype=np.float64) + epsilon * direction
        minus_array = np.asarray(base, dtype=np.float64) - epsilon * direction
        setter(plus_array)
        plus = float(objective())
        setter(minus_array)
        minus = float(objective())
        setter(np.asarray(base, dtype=np.float64))
    else:
        plus = minus = float("nan")
    observed = abs(plus - minus) if np.isfinite(plus) and np.isfinite(minus) else float("nan")
    resolution = float(np.finfo(dtype).eps) * max(
        abs(plus) if np.isfinite(plus) else 0.0,
        abs(minus) if np.isfinite(minus) else 0.0,
        1e-12,
    )
    ratio = float("inf") if resolution == 0.0 else observed / resolution
    return ResolutionProbe(
        expected_signal=expected,
        observed_signal=float(observed),
        dtype_resolution=resolution,
        signal_to_resolution=ratio,
        resolvable=bool(np.isfinite(ratio) and ratio >= resolvability_ratio),
    )


def sweep_epsilon(
    objective_factory: Callable[[], float],
    parameters: Sequence[Array],
    analytic: Sequence[float],
    epsilons: Sequence[float],
    atol: float,
    rtol: float,
) -> List[FiniteDifferenceResult]:
    """Finite differences over several step sizes.

    Finite differences trade truncation against round-off error, so a single
    epsilon is not evidence. This records the whole sweep and lets a caller
    report the achievable error floor instead of one lucky value.
    """
    analytic_flat = np.concatenate(
        [as_float_array(p).ravel() for p in parameters]
    ) if parameters else np.zeros(0)
    results: List[FiniteDifferenceResult] = []
    for epsilon in epsilons:
        chunks = [central_difference(objective_factory, np.asarray(p, dtype=np.float64), epsilon) for p in parameters]
        numeric = np.concatenate([c.ravel() for c in chunks]) if chunks else np.zeros(0)
        results.append(
            compare_gradients(analytic_flat, numeric, epsilon, atol, rtol)
        )
    return results


def truncation_noise_floor(results: Sequence[FiniteDifferenceResult]) -> Optional[float]:
    """Smallest observed relative L2 error across an epsilon sweep."""
    values = [r.relative_l2_error for r in results if np.isfinite(r.relative_l2_error)]
    return float(min(values)) if values else None


def finite_difference_stability(
    results: Sequence[FiniteDifferenceResult],
) -> Dict[str, object]:
    """Summarise how much the epsilon choice changes the numeric gradient."""
    rel = [r.relative_l2_error for r in results if np.isfinite(r.relative_l2_error)]
    if len(rel) < 2:
        return {"n_epsilons": len(rel), "spread_relative_l2": None}
    return {
        "n_epsilons": len(rel),
        "spread_relative_l2": float(max(rel) - min(rel)),
        "min_relative_l2": float(min(rel)),
        "max_relative_l2": float(max(rel)),
        "epsilons": [r.epsilon for r in results],
        "note": (
            "Spread across epsilon values is a property of the finite-difference "
            "estimator, not of the implementation under test."
        ),
    }


# ---------------------------------------------------------------------------
# independent reference objectives (used to check the forward pass itself)
# ---------------------------------------------------------------------------
def reference_mse(y_true: Array, y_pred: Array) -> float:
    """Mean of 0.5*(y_true - y_pred)^2, matching the documented dense contract."""
    diff = as_float_array(y_true).ravel() - as_float_array(y_pred).ravel()
    return float(np.mean(0.5 * diff ** 2))


def reference_bce(y_true: Array, y_pred: Array, eps: float = 1e-7) -> float:
    """Clipped mean binary cross-entropy, matching the declared objective."""
    p = np.clip(as_float_array(y_pred).ravel(), eps, 1.0 - eps)
    t = np.clip(as_float_array(y_true).ravel(), eps, 1.0 - eps)
    return float(np.mean(-(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))))


def reference_objective(y_true: Array, y_pred: Array, loss_fn: str) -> float:
    key = (loss_fn or "bce").lower()
    if key == "mse":
        return reference_mse(y_true, y_pred)
    return reference_bce(y_true, y_pred)


def declared_objective_kind(loss_fn: str) -> str:
    return "mse" if (loss_fn or "bce").lower() == "mse" else "bce"


def is_bce_with_sigmoid_contract(loss_fn: str, output_activation: Optional[str]) -> bool:
    """Whether the fused BCE/sigmoid gradient contract is declared applicable.

    The dense backward engine only implements the fused derivative for a
    sigmoid output. Any other combination falls through to ``delta = y_hat - y``
    which is not the BCE derivative, so those combinations must be reported
    separately instead of being pooled with validated cases.
    """
    key = (loss_fn or "bce").lower()
    act = (output_activation or "linear").lower()
    return key == "bce" and act == "sigmoid"


def effective_batch_size(input_vec: Array) -> int:
    """Number of leading examples the implementation's mean reduction divides by."""
    arr = as_float_array(input_vec)
    if arr.ndim <= 1:
        return max(1, arr.size)
    return max(1, int(arr.shape[0]))


NON_SMOOTH_ACTIVATIONS = {"relu", "leaky_relu"}


def kink_band_analysis(graph, band: float) -> Dict[str, object]:
    """Measure how much of a network sits inside a non-differentiable kink band.

    Central differences are invalid wherever the function being differentiated
    is not locally linear. For ReLU-like activations the affected region is
    ``|pre-activation| <= band`` for a perturbation of size ``band``. Reporting
    the occupied fraction converts an unexplained gradient mismatch into a
    measurable statement about the estimator.
    """
    entries: List[Dict[str, object]] = []
    total = 0
    within = 0
    for index, layer in enumerate(getattr(graph, "layer_instances", []) or []):
        pre = getattr(layer, "Z", None)
        activation = str(getattr(layer, "activation_name", None) or getattr(layer, "cfg", None) and getattr(layer.cfg, "activation", "") or "")
        activation = activation.lower()
        if pre is None or activation not in NON_SMOOTH_ACTIVATIONS:
            continue
        arr = as_float_array(pre)
        count = int(np.count_nonzero(np.abs(arr) <= band))
        total += int(arr.size)
        within += count
        entries.append(
            {
                "layer_instance_index": index,
                "layer_type": getattr(layer, "layer_type", "unknown"),
                "activation": activation,
                "pre_activation_size": int(arr.size),
                "pre_activation_min_abs": float(np.min(np.abs(arr))) if arr.size else None,
                "within_band": count,
            }
        )
    return {
        "band": band,
        "layers_with_non_smooth_activation": len(entries),
        "pre_activation_elements_examined": total,
        "elements_within_band": within,
        "fraction_within_band": (within / total) if total else 0.0,
        "details": entries,
        "note": (
            "If fraction_within_band > 0 the finite-difference gradient of every "
            "parameter upstream of those activations is biased, and a mismatch "
            "with the analytic gradient is evidence about the estimator."
        ),
    }


def finite(value: float) -> bool:
    return bool(math.isfinite(float(value)))