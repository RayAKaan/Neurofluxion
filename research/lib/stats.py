"""Descriptive and inferential statistics helpers for research experiments.

The repository must not manufacture significance. These helpers therefore make
the sample size, the dispersion and the estimator explicit for every statistic
they return, and they refuse to compute a normal-approximation confidence
interval when the sample is too small for that approximation to be meaningful.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

# Below this sample size a t-based interval is not reported.
MIN_SAMPLES_FOR_CI = 5


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def describe(values: Sequence[float]) -> Dict[str, object]:
    """Full descriptive summary of a sample, preserving count of outliers."""
    raw = np.asarray(list(values), dtype=np.float64)
    n_total = int(raw.size)
    finite = _finite(values)
    if finite.size == 0:
        return {
            "n_total": n_total,
            "n_finite": 0,
            "n_nonfinite": n_total,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {},
            "confidence_interval_95": None,
            "note": "no finite samples",
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    median = float(np.median(finite))
    q1, q3 = (float(v) for v in np.percentile(finite, [25, 75]))
    iqr = q3 - q1
    outlier_mask = np.zeros(finite.shape, dtype=bool)
    if iqr > 0 and finite.size > 3:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (finite < lower) | (finite > upper)
    percentiles = {
        "p05": float(np.percentile(finite, 5)),
        "p25": q1,
        "p50": median,
        "p75": q3,
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
    }
    return {
        "n_total": n_total,
        "n_finite": int(finite.size),
        "n_nonfinite": n_total - int(finite.size),
        "mean": mean,
        "median": median,
        "std": std,
        "cv": (std / mean) if mean not in (0.0,) else None,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "percentiles": percentiles,
        "iqr": iqr,
        "n_outliers_iqr": int(np.count_nonzero(outlier_mask)),
        "outlier_values": [float(v) for v in finite[outlier_mask]],
        "confidence_interval_95_mean": _ci95_mean(finite),
        "confidence_interval_95_median_note": (
            "A normal-approximation interval is reported for the mean only. "
            "Non-parametric medians are reported without an inferential interval."
        ),
    }


def _ci95_mean(sample: np.ndarray) -> Optional[Dict[str, float]]:
    n = int(sample.size)
    if n < MIN_SAMPLES_FOR_CI:
        return None
    mean = float(np.mean(sample))
    sd = float(np.std(sample, ddof=1))
    if sd == 0.0:
        return {"low": mean, "high": mean, "method": "degenerate_zero_variance"}
    try:
        from scipy import stats  # type: ignore

        half = float(stats.t.ppf(0.975, df=n - 1) * sd / math.sqrt(n))
        method = "student_t_95"
    except Exception:
        # Normal approximation, explicitly labelled.
        half = 1.959963985 * sd / math.sqrt(n)
        method = "normal_approx_95_no_scipy"
    return {"low": mean - half, "high": mean + half, "method": method}


def ratio_stats(
    numerator: Sequence[float],
    denominator: Sequence[float],
) -> Dict[str, object]:
    """Statistics for per-trial ratios paired by index.

    Only paired ratios are computed. Ratio-of-medians is also reported, but the
    distribution summary comes from the paired per-trial ratios because a ratio
    of independently summarised statistics has no sampling distribution.
    """
    num = np.asarray(list(numerator), dtype=np.float64)
    den = np.asarray(list(denominator), dtype=np.float64)
    if num.size != den.size:
        raise ValueError("ratio_stats requires paired samples of equal length")
    if num.size == 0:
        return {"n_pairs": 0, "paired_ratio": describe([])}
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0)
    paired = num[valid] / den[valid]
    median_den = float(np.median(den[valid])) if valid.any() else None
    median_num = float(np.median(num[valid])) if valid.any() else None
    return {
        "n_pairs": int(num.size),
        "n_valid_pairs": int(valid.sum()),
        "paired_ratio": describe(paired.tolist()),
        "ratio_of_medians": (
            median_num / median_den if median_den not in (None, 0.0) else None
        ),
        "method_note": (
            "paired_ratio uses per-trial ratios; ratio_of_medians is a descriptive "
            "point estimate only and has no inferential interval."
        ),
    }


def effect_size_paired(a: Sequence[float], b: Sequence[float]) -> Optional[Dict[str, float]]:
    """Cohen's d_z for paired samples, with the sample size that justifies it."""
    arr_a = _finite(a)
    arr_b = _finite(b)
    if arr_a.size != arr_b.size or arr_a.size < 3:
        return None
    diff = arr_a - arr_b
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return {"cohens_dz": None, "mean_difference": float(np.mean(diff)), "sd_difference": 0.0}
    return {
        "cohens_dz": float(np.mean(diff) / sd),
        "mean_difference": float(np.mean(diff)),
        "sd_difference": sd,
        "n_pairs": int(arr_a.size),
    }


def mann_whitney_p(
    a: Sequence[float],
    b: Sequence[float],
    min_samples: int = 8,
) -> Optional[Dict[str, object]]:
    """Two-sided Mann-Whitney U p-value, refused for undersized samples.

    Returns ``None`` when the sample is too small or SciPy is unavailable so a
    caller cannot accidentally report a p-value as evidence.
    """
    arr_a = _finite(a)
    arr_b = _finite(b)
    if arr_a.size < min_samples or arr_b.size < min_samples:
        return None
    try:
        from scipy import stats  # type: ignore

        result = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    except Exception:
        return None
    return {
        "p_value": float(result.pvalue),
        "u_statistic": float(result.statistic),
        "n_a": int(arr_a.size),
        "n_b": int(arr_b.size),
        "min_samples_required": min_samples,
        "interpretation_guard": (
            "A p-value is not evidence of practical importance and does not "
            "substitute for the reported effect size and dispersion."
        ),
    }


def percentile_of(values: Sequence[float], threshold: float) -> Optional[float]:
    finite = _finite(values)
    if finite.size == 0:
        return None
    return float(np.mean(finite <= threshold))


def summarise_matrix(values: np.ndarray) -> Dict[str, object]:
    """Elementwise summary of an error matrix, used for gradient comparisons."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def flat_summary(values: Sequence[float], keys: Optional[List[str]] = None) -> Dict[str, object]:
    """Compact ``key -> statistic`` mapping for CSV table export."""
    summary = describe(values)
    if keys is None:
        keys = ["mean", "median", "std", "min", "max"]
    return {key: summary.get(key) for key in keys}