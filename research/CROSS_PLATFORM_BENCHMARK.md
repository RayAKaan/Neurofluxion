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

## First cross-platform run (2026-10-14)

Workflow run [38571861118](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571861118) passed on both OS runners. Both artifacts recorded tested checkout revision `7c82d865f9b895d95e63b6e54508badaa7f9a320`, NumPy 2.4.6, and 40 samples per path/configuration. Output parity passed with maximum absolute error 0.0 for all four cases on both systems.

| Widths (input-hidden-output) | Linux forward median (ms) | Linux trace median (ms) | Linux ratio | Windows forward median (ms) | Windows trace median (ms) | Windows ratio |
|---|---:|---:|---:|---:|---:|---:|
| [4, 8, 2] | 0.005938 | 0.013466 | 2.27× | 0.007950 | 0.017000 | 2.14× |
| [16, 32, 8] | 0.006074 | 0.028488 | 4.69× | 0.008200 | 0.034550 | 4.21× |
| [32, 64, 16] | 0.006225 | 0.065103 | 10.46× | 0.008500 | 0.071100 | 8.36× |
| [64, 128, 32] | 0.006620 | 0.201167 | 30.39× | 0.009200 | 0.221550 | 24.08× |

Environment metadata:
- Linux runner: Ubuntu/Linux 6.17, x86_64, Python 3.11.17, NumPy 2.4.6.
- Windows runner: Windows 10 runner image, AMD64, Python 3.11.9, NumPy 2.4.6.

The ratio increases with graph width in both environments, while the absolute timing differs by OS and runner. This is evidence that the qualitative pattern reproduces across these two hosted runner environments, not evidence of universal overhead: there is one run per OS, both are CI runners, and full trace creation performs more work than returning the final output. Re-run independent workflow instances and add a local/hardware-diverse environment before drawing stronger conclusions.

Artifacts: [Linux report](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571861118/artifacts/11897078940) and [Windows report](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571861118/artifacts/11898462192). Each contains raw samples and environment/source metadata.
