from __future__ import annotations

import numpy as np


class BatchNormLayer:
    layer_type = "batchnorm"
    has_params = True

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.x_hat: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.dgamma: np.ndarray | None = None
        self.dbeta: np.ndarray | None = None
        self.last_input: np.ndarray | None = None

    def _compute_stats(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x.ndim == 1:
            mean = x
            var = np.zeros_like(x)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            return mean, var, x_hat
        if x.ndim == 3:
            # per-channel stats over spatial dims
            mean = x.mean(axis=(1, 2))
            var = x.var(axis=(1, 2))
            x_hat = (x - mean[:, None, None]) / np.sqrt(var[:, None, None] + self.eps)
            return mean, var, x_hat
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return mean, var, x_hat

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        self.last_input = x
        mean, var, x_hat = self._compute_stats(x)
        self.mean = mean
        self.var = var
        self.x_hat = x_hat
        # update running stats
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        if x.ndim == 3:
            return self.gamma[:, None, None] * x_hat + self.beta[:, None, None]
        return self.gamma * x_hat + self.beta

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.x_hat is None or self.last_input is None or self.mean is None or self.var is None:
            raise RuntimeError("BatchNormLayer.backward called before forward.")
        x = self.last_input
        if x.ndim == 3:
            c, h, w = x.shape
            N = h * w
            x_hat = self.x_hat
            self.dgamma = np.sum(d_out * x_hat, axis=(1, 2))
            self.dbeta = np.sum(d_out, axis=(1, 2))
            dx_hat = d_out * self.gamma[:, None, None]
            dvar = np.sum(dx_hat * (x - self.mean[:, None, None]) * -0.5 * (self.var[:, None, None] + self.eps) ** -1.5, axis=(1, 2))
            dmean = np.sum(dx_hat * -1 / np.sqrt(self.var[:, None, None] + self.eps), axis=(1, 2)) + dvar * np.mean(-2 * (x - self.mean[:, None, None]), axis=(1, 2))
            dx = dx_hat / np.sqrt(self.var[:, None, None] + self.eps) + dvar[:, None, None] * 2 * (x - self.mean[:, None, None]) / N + dmean[:, None, None] / N
            return dx.astype(np.float32)

        x_hat = self.x_hat
        if x.ndim == 1:
            # A single observation has zero centered activation under the
            # layer's current single-sample statistics.
            self.dgamma = np.zeros_like(self.gamma)
            self.dbeta = np.asarray(d_out, dtype=np.float32).copy()
            return np.zeros_like(x, dtype=np.float32)

        N = x.shape[0]
        self.dgamma = np.sum(d_out * x_hat, axis=0)
        self.dbeta = np.sum(d_out, axis=0)
        dx_hat = d_out * self.gamma
        dvar = np.sum(
            dx_hat * (x - self.mean) * -0.5 * (self.var + self.eps) ** -1.5,
            axis=0,
        )
        dmean = (
            np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=0)
            + dvar * np.mean(-2 * (x - self.mean), axis=0)
        )
        dx = (
            dx_hat / np.sqrt(self.var + self.eps)
            + dvar * 2 * (x - self.mean) / N
            + dmean / N
        )
        return dx.astype(np.float32)

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.gamma, self.beta

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dgamma, self.dbeta

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        self.gamma = w.astype(np.float32)
        self.beta = b.astype(np.float32)

