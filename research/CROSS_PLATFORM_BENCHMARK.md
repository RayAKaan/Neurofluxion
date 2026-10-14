# Cross-Platform Trace Benchmark

## Purpose

Compare the existing NumPy simulator's final-output forward path with its full dense trace-construction path on two operating-system runner families. The goal is to characterize whether the previously observed ratios are specific to one hosted-runner environment, not to claim a universal tracing overhead.

## Protocol

- Fixed source revision per workflow run; the report records the exact checkout commit.
- Ubuntu and Windows GitHub-hosted runners, Python 3.11, and the installed NumPy version recorded in each artifact.
- Four fixed dense widths: [4, 8, 2], [16, 32, 8], [32, 64, 16], and [64, 128, 32].
- Fixed seeded initialization and deterministic input per graph configuration.
- Five warmups per path/configuration and 40 raw timing samples per path/configuration.
- Check final-output parity before timing and record max absolute error.
- Preserve all samples, medians, p90, min/max, machine/processor metadata, commit SHA, and tolerance in JSON artifacts.
- Run both paths sequentially on the same runner; do not compare absolute milliseconds as if runners were identical.

## Commands

```bash
PYTHONPATH=backend python research/experiments/run_trace_benchmark.py --repo . --out research/artifacts/trace-benchmark.json
```

The workflow matrix is defined in .github/workflows/cross-platform-trace-benchmark.yml and uploads one report per OS.

## Interpretation limits

The trace path performs additional work: it constructs operation records and parameter snapshots. The ratio is descriptive of these two implementations, not an estimate of an abstract trace feature's causal cost. Ubuntu and Windows are different environments, but both are hosted CI systems and do not replace a local workstation or a second hardware class. A benchmark report passing means output parity held and measurements were recorded; it does not establish performance superiority or production latency.
