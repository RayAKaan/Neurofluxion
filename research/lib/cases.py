"""Canonical registry of Neurofluxion simulator graph cases.

One registry is shared by numerical validation, training-update validation,
trace-fidelity auditing and benchmarking so that a case identifier always refers
to the same architecture, input distribution, objective and tolerance policy.

A case is a *fixed experimental configuration*. Nothing here changes runtime
behaviour; the module only constructs ``LayerConfig`` lists and deterministic
inputs for the pinned simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from simulator.layers import LayerConfig

Array = np.ndarray


@dataclass(frozen=True)
class GraphCase:
    """A fixed, reproducible simulator graph configuration."""

    case_id: str
    family: str
    description: str
    layers: Sequence[LayerConfig]
    loss_fn: str
    seed: int
    gradient_tolerance: Dict[str, float] = field(
        default_factory=lambda: {"atol": 2e-3, "rtol": 2e-2}
    )
    gradient_relative_l2_max: float = 5e-2
    gradient_norm_floor: float = 1e-6
    forward_tolerance: Dict[str, float] = field(
        default_factory=lambda: {"atol": 1e-5, "rtol": 1e-5}
    )
    supports_finite_difference: bool = True
    supports_training_update: bool = False
    expectation: str = "supported"
    notes: str = ""
    _input_factory: Optional[Callable[[], Array]] = field(default=None, repr=False)
    _target_factory: Optional[Callable[[], Array]] = field(default=None, repr=False)

    # -- inputs ------------------------------------------------------------
    def make_input(self) -> Array:
        if self._input_factory is None:
            raise ValueError(f"case '{self.case_id}' has no input factory")
        rng = np.random.default_rng(self.seed)
        return self._input_factory(rng).astype(np.float32)

    def make_target(self, output_size: int) -> Array:
        if self._target_factory is None:
            rng = np.random.default_rng(self.seed + 977)
            return rng.uniform(0.2, 0.8, size=output_size).astype(np.float32)
        return self._target_factory().astype(np.float32)

    def layer_summary(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for layer in self.layers:
            out.append(
                {
                    "layer_type": layer.layer_type,
                    "neurons": layer.neurons,
                    "activation": layer.activation,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "filters": layer.filters,
                    "pool_size": layer.pool_size,
                    "hidden_size": layer.hidden_size,
                    "return_sequences": layer.return_sequences,
                    "embedding_dim": layer.embedding_dim,
                    "vocab_size": layer.vocab_size,
                    "num_heads": layer.num_heads,
                    "input_shape": layer.input_shape,
                }
            )
        return out


def _linear_input(width: int) -> Callable[[np.random.Generator], Array]:
    def factory(rng: np.random.Generator) -> Array:
        return np.linspace(-0.7, 0.7, width, dtype=np.float32)

    return factory


def _uniform_input(shape) -> Callable[[np.random.Generator], Array]:
    def factory(rng: np.random.Generator) -> Array:
        return rng.uniform(-0.8, 0.8, size=shape).astype(np.float32)

    return factory


def _token_input(tokens: Sequence[int]) -> Callable[[np.random.Generator], Array]:
    def factory(rng: np.random.Generator) -> Array:
        return np.asarray(tokens, dtype=np.float32)

    return factory


def _binary_target(size: int, value: float) -> Callable[[], Array]:
    def factory() -> Array:
        return np.full(size, value, dtype=np.float32)

    return factory


# ---------------------------------------------------------------------------
# dense family
# ---------------------------------------------------------------------------
DENSE_RELU_RELU_LINEAR = GraphCase(
    case_id="dense_3_4_2_relu",
    family="dense",
    description="Input(3) -> Dense(4, relu) -> Output(2, linear), mean-reduced MSE",
    layers=(
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("dense", neurons=4, activation="relu"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=1001,
    supports_training_update=True,
    notes="Two-hidden-layer-free dense stack with a non-smooth ReLU; the "
    "epsilon sweep documents finite-difference sensitivity at the kink.",
    _input_factory=_linear_input(3),
)

DENSE_TANH_LINEAR_MSE = GraphCase(
    case_id="dense_4_5_1_tanh_linear_mse",
    family="dense",
    description="Input(4) -> Dense(5, tanh) -> Dense(3, leaky_relu) -> Output(1, linear), MSE",
    layers=(
        LayerConfig("input", neurons=4, input_shape=[4]),
        LayerConfig("dense", neurons=5, activation="tanh"),
        LayerConfig("dense", neurons=3, activation="leaky_relu"),
        LayerConfig("output", neurons=1, activation="linear"),
    ),
    loss_fn="mse",
    seed=1002,
    supports_training_update=True,
    _input_factory=_linear_input(4),
)

DENSE_SIGMOID_BCE = GraphCase(
    case_id="dense_3_4_1_sigmoid_bce",
    family="dense",
    description="Input(3) -> Dense(4, tanh) -> Output(1, sigmoid), binary cross-entropy",
    layers=(
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("dense", neurons=4, activation="tanh"),
        LayerConfig("output", neurons=1, activation="sigmoid"),
    ),
    loss_fn="bce",
    seed=1003,
    gradient_tolerance={"atol": 2e-3, "rtol": 2e-2},
    supports_training_update=True,
    _input_factory=_linear_input(3),
    _target_factory=_binary_target(1, 1.0),
)

DENSE_DEEP_MSE = GraphCase(
    case_id="dense_deep_6x4_mse",
    family="dense",
    description="Input(5) -> four tanh dense layers -> Output(3, linear), MSE",
    layers=(
        LayerConfig("input", neurons=5, input_shape=[5]),
        LayerConfig("dense", neurons=6, activation="tanh"),
        LayerConfig("dense", neurons=6, activation="tanh"),
        LayerConfig("dense", neurons=4, activation="tanh"),
        LayerConfig("dense", neurons=4, activation="tanh"),
        LayerConfig("output", neurons=3, activation="linear"),
    ),
    loss_fn="mse",
    seed=1004,
    supports_training_update=True,
    notes="Deeper stack used to check accumulation of gradient error over layers.",
    _input_factory=_uniform_input(5),
)

DENSE_MULTI_BATCH_MSE = GraphCase(
    case_id="dense_3_4_2_mse_batch3",
    family="dense",
    description="Input(3) -> Dense(4, tanh) -> Output(2, linear) evaluated on a batch of three rows, MSE",
    layers=(
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("dense", neurons=4, activation="tanh"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=1005,
    supports_finite_difference=False,
    supports_training_update=False,
    expectation="unsupported",
    notes="Recorded as an explicit capability boundary: the dense layer flattens "
    "its input with reshape(-1), so a (3, 3) batch cannot be executed. A batch "
    "dimension is therefore not supported on the dense path.",
    _input_factory=_uniform_input((3, 3)),
)


# ---------------------------------------------------------------------------
# convolutional family
# ---------------------------------------------------------------------------
CONV_FLATTEN_DENSE = GraphCase(
    case_id="conv1_5x5_1_valid_flatten_dense",
    family="conv",
    description="Input(1,5,5) -> Conv2D(1 filter, 3x3, valid, relu) -> Flatten -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[1, 5, 5]),
        LayerConfig("conv2d", filters=1, kernel_size=3, stride=1, padding="valid", activation="relu"),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=2001,
    supports_training_update=True,
    _input_factory=_uniform_input((1, 5, 5)),
)

CONV_SAME_STRIDE2_BN = GraphCase(
    case_id="conv1_6x6_2_same_s2_batchnorm_flatten_dense",
    family="conv",
    description="Input(1,6,6) -> Conv2D(2 filters, 3x3, same, stride 2, relu) -> BatchNorm -> Flatten -> Output(1, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[1, 6, 6]),
        LayerConfig(
            "conv2d", filters=2, kernel_size=3, stride=2, padding="same", activation="relu"
        ),
        LayerConfig("batchnorm"),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=1, activation="linear"),
    ),
    loss_fn="mse",
    seed=2002,
    supports_training_update=True,
    notes="Exercises 'same' padding, stride 2, multi-channel BatchNorm and the "
    "spatial BatchNorm backward path.",
    _input_factory=_uniform_input((1, 6, 6)),
)

CONV_POOL_FLATTEN_DENSE = GraphCase(
    case_id="conv_pool_max_flatten_dense",
    family="conv",
    description="Input(1,6,6) -> Conv2D(1,3x3,valid,relu) -> MaxPool2D(2) -> Flatten -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[1, 6, 6]),
        LayerConfig("conv2d", filters=1, kernel_size=3, stride=1, padding="valid", activation="relu"),
        LayerConfig("maxpool2d", pool_size=2),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=2003,
    supports_training_update=True,
    _input_factory=_uniform_input((1, 6, 6)),
)

CONV_AVGPOOL_FLATTEN_DENSE = GraphCase(
    case_id="conv_pool_avg_flatten_dense",
    family="conv",
    description="Input(1,6,6) -> Conv2D(1,3x3,valid,relu) -> AvgPool2D(2) -> Flatten -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[1, 6, 6]),
        LayerConfig("conv2d", filters=1, kernel_size=3, stride=1, padding="valid", activation="relu"),
        LayerConfig("avgpool2d", pool_size=2),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=2004,
    supports_training_update=True,
    _input_factory=_uniform_input((1, 6, 6)),
)


CONV_SAME_STRIDE2_BN_TANH = GraphCase(
    case_id="conv1_6x6_2_same_s2_batchnorm_flatten_tanh",
    family="conv",
    description="Input(1,6,6) -> Conv2D(2 filters, 3x3, same, stride 2, tanh) -> BatchNorm -> Flatten -> Output(1, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[1, 6, 6]),
        LayerConfig(
            "conv2d", filters=2, kernel_size=3, stride=2, padding="same", activation="tanh"
        ),
        LayerConfig("batchnorm"),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=1, activation="linear"),
    ),
    loss_fn="mse",
    seed=2005,
    supports_training_update=True,
    gradient_relative_l2_max=2e-2,
    notes="Smooth-activation control for conv1_6x6_2_same_s2_batchnorm_flatten_dense. "
    "The ReLU variant cannot be validated by central differences because a large "
    "fraction of its pre-activations lie inside the kink band.",
    _input_factory=_uniform_input((1, 6, 6)),
)


# ---------------------------------------------------------------------------
# normalisation family
# ---------------------------------------------------------------------------
BATCHNORM_VECTOR_DENSE = GraphCase(
    case_id="batchnorm_vector_5_dense",
    family="normalization",
    description="Input(5) -> BatchNorm(5) -> Output(2, linear), MSE (degenerate single-sample normalisation)",
    layers=(
        LayerConfig("input", neurons=5, input_shape=[5]),
        LayerConfig("batchnorm"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=3001,
    supports_training_update=True,
    notes="With a one-dimensional input the layer's per-feature mean equals the "
    "feature itself and the variance is zero, so the normalised output is "
    "beta regardless of the input. Reported as a documented degeneracy.",
    _input_factory=_uniform_input(5),
)

BATCHNORM_SPATIAL_DENSE = GraphCase(
    case_id="batchnorm_spatial_conv",
    family="normalization",
    description="Input(2,4,4) -> Conv2D(2,3x3,valid,linear) -> BatchNorm -> Flatten -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[2, 4, 4]),
        LayerConfig("conv2d", filters=2, kernel_size=3, stride=1, padding="valid", activation="linear"),
        LayerConfig("batchnorm"),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=3002,
    supports_training_update=True,
    notes="Non-degenerate BatchNorm: per-channel statistics over a 2x2 spatial map.",
    _input_factory=_uniform_input((2, 4, 4)),
)


# ---------------------------------------------------------------------------
# recurrent family
# ---------------------------------------------------------------------------
RNN_DENSE = GraphCase(
    case_id="rnn_t3_d2_h3_dense",
    family="recurrent",
    description="Input(T=3,D=2) -> RNN(hidden 3, tanh) -> Output(1, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[3, 2]),
        LayerConfig("rnn", hidden_size=3, return_sequences=False),
        LayerConfig("output", neurons=1, activation="linear"),
    ),
    loss_fn="mse",
    seed=4001,
    supports_training_update=True,
    _input_factory=_uniform_input((3, 2)),
)

LSTM_DENSE = GraphCase(
    case_id="lstm_t4_d3_h3_dense",
    family="recurrent",
    description="Input(T=4,D=3) -> LSTM(hidden 3) -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[4, 3]),
        LayerConfig("lstm", hidden_size=3, return_sequences=False),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=4002,
    supports_training_update=True,
    _input_factory=_uniform_input((4, 3)),
)

GRU_DENSE = GraphCase(
    case_id="gru_t4_d3_h3_dense",
    family="recurrent",
    description="Input(T=4,D=3) -> GRU(hidden 3) -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[4, 3]),
        LayerConfig("gru", hidden_size=3, return_sequences=False),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=4003,
    supports_training_update=True,
    _input_factory=_uniform_input((4, 3)),
)

LSTM_RETURN_SEQUENCES_DENSE = GraphCase(
    case_id="lstm_t3_d2_h2_returnseq_dense",
    family="recurrent",
    description="Input(T=3,D=2) -> LSTM(hidden 2, return_sequences) -> Flatten -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[3, 2]),
        LayerConfig("lstm", hidden_size=2, return_sequences=True),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=4004,
    supports_training_update=True,
    notes="return_sequences keeps the sequence shape and consumes it via Flatten.",
    _input_factory=_uniform_input((3, 2)),
)


# ---------------------------------------------------------------------------
# embedding / attention family
# ---------------------------------------------------------------------------
EMBEDDING_ATTENTION_DENSE = GraphCase(
    case_id="embedding_attention_rnn_dense",
    family="embedding_attention",
    description="Token input [1,2,1] -> Embedding(vocab 5, dim 2) -> Attention(1 head) -> RNN(hidden 2) -> Output(1, sigmoid), BCE",
    layers=(
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("embedding", embedding_dim=2, vocab_size=5, neurons=2),
        LayerConfig("attention", num_heads=1, neurons=2),
        LayerConfig("rnn", hidden_size=2, return_sequences=False),
        LayerConfig("output", neurons=1, activation="sigmoid"),
    ),
    loss_fn="bce",
    seed=5001,
    gradient_tolerance={"atol": 3e-3, "rtol": 3e-2},
    gradient_relative_l2_max=2e-1,
    gradient_norm_floor=1e-6,
    supports_training_update=True,
    notes="Repeated token index 1 exercises accumulation into a shared embedding row.",
    _input_factory=_token_input((1, 2, 1)),
    _target_factory=_binary_target(1, 1.0),
)

EMBEDDING_ATTENTION_MULTIHEAD = GraphCase(
    case_id="embedding_attention_2head_dense",
    family="embedding_attention",
    description="Token input [0,3,3] -> Embedding(vocab 4, dim 4) -> Attention(num_heads=2) -> Flatten -> Output(2, linear), MSE",
    layers=(
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("embedding", embedding_dim=4, vocab_size=4, neurons=4),
        LayerConfig("attention", num_heads=2, neurons=4),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=5002,
    notes="Requests two attention heads. The single-head claim for this case is "
    "an audited finding, not an assumption.",
    _input_factory=_token_input((0, 3, 3)),
)


# ---------------------------------------------------------------------------
# residual family
# ---------------------------------------------------------------------------
RESIDUAL_DENSE = GraphCase(
    case_id="residual_dense_4_4",
    family="residual",
    description="Input(4) -> Dense(4, linear) -> Residual -> Dense(2, linear), MSE",
    layers=(
        LayerConfig("input", neurons=4, input_shape=[4]),
        LayerConfig("dense", neurons=4, activation="linear"),
        LayerConfig("residual"),
        LayerConfig("output", neurons=2, activation="linear"),
    ),
    loss_fn="mse",
    seed=6001,
    supports_training_update=True,
    _input_factory=_uniform_input(4),
)


# ---------------------------------------------------------------------------
# composite family
# ---------------------------------------------------------------------------
COMPOSITE_CONV_LSTM_DENSE = GraphCase(
    case_id="composite_conv_pool_dense_residual",
    family="composite",
    description="Input(1,6,6) -> Conv2D(1,3x3,valid,relu) -> MaxPool2D(2) -> Flatten -> Dense(3, tanh) -> Residual -> Output(1, linear), MSE",
    layers=(
        LayerConfig("input", input_shape=[1, 6, 6]),
        LayerConfig("conv2d", filters=1, kernel_size=3, stride=1, padding="valid", activation="relu"),
        LayerConfig("maxpool2d", pool_size=2),
        LayerConfig("flatten"),
        LayerConfig("dense", neurons=3, activation="tanh"),
        LayerConfig("residual"),
        LayerConfig("output", neurons=1, activation="linear"),
    ),
    loss_fn="mse",
    seed=7001,
    supports_training_update=True,
    notes="Cross-family composition: spatial features feed two dense stages with "
    "an identity residual connection between them.",
    _input_factory=_uniform_input((1, 6, 6)),
)

COMPOSITE_EMBEDDING_ATTENTION_LSTM = GraphCase(
    case_id="composite_embedding_attention_lstm",
    family="composite",
    description="Token input [2,0,1] -> Embedding(vocab 4, dim 2) -> Attention(1 head) -> LSTM(hidden 2) -> Output(1, linear), MSE",
    layers=(
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("embedding", embedding_dim=2, vocab_size=4, neurons=2),
        LayerConfig("attention", num_heads=1, neurons=2),
        LayerConfig("lstm", hidden_size=2, return_sequences=False),
        LayerConfig("output", neurons=1, activation="linear"),
    ),
    loss_fn="mse",
    seed=7002,
    supports_training_update=True,
    notes="Embedding and attention feed a recurrent layer rather than a plain "
    "recurrent stack, so the packed-parameter layout differs across families.",
    _input_factory=_token_input((2, 0, 1)),
)


ALL_CASES: Dict[str, GraphCase] = {
    case.case_id: case
    for case in (
        DENSE_RELU_RELU_LINEAR,
        DENSE_TANH_LINEAR_MSE,
        DENSE_SIGMOID_BCE,
        DENSE_DEEP_MSE,
        DENSE_MULTI_BATCH_MSE,
        CONV_FLATTEN_DENSE,
        CONV_SAME_STRIDE2_BN,
        CONV_SAME_STRIDE2_BN_TANH,
        CONV_POOL_FLATTEN_DENSE,
        CONV_AVGPOOL_FLATTEN_DENSE,
        BATCHNORM_VECTOR_DENSE,
        BATCHNORM_SPATIAL_DENSE,
        RNN_DENSE,
        LSTM_DENSE,
        GRU_DENSE,
        LSTM_RETURN_SEQUENCES_DENSE,
        EMBEDDING_ATTENTION_DENSE,
        EMBEDDING_ATTENTION_MULTIHEAD,
        RESIDUAL_DENSE,
        COMPOSITE_CONV_LSTM_DENSE,
        COMPOSITE_EMBEDDING_ATTENTION_LSTM,
    )
}


def get_case(case_id: str) -> GraphCase:
    if case_id not in ALL_CASES:
        raise KeyError(f"unknown graph case '{case_id}'")
    return ALL_CASES[case_id]


def select_cases(
    case_ids: Optional[Sequence[str]] = None,
    family: Optional[str] = None,
) -> List[GraphCase]:
    selected = list(ALL_CASES.values())
    if family:
        selected = [c for c in selected if c.family == family]
    if case_ids:
        wanted = list(case_ids)
        selected = [c for c in selected if c.case_id in wanted]
    return selected


def case_index() -> List[Dict[str, object]]:
    return [
        {
            "case_id": case.case_id,
            "family": case.family,
            "description": case.description,
            "loss_fn": case.loss_fn,
            "seed": case.seed,
            "supports_finite_difference": case.supports_finite_difference,
            "supports_training_update": case.supports_training_update,
            "expectation": case.expectation,
            "gradient_atol": case.gradient_tolerance["atol"],
            "gradient_rtol": case.gradient_tolerance["rtol"],
            "forward_atol": case.forward_tolerance["atol"],
            "forward_rtol": case.forward_tolerance["rtol"],
            "notes": case.notes,
        }
        for case in ALL_CASES.values()
    ]