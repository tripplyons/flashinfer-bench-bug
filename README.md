# FlashInfer-Bench Bug Repro

This repo benchmarks the official `gdn_prefill` example solution from the MLSys 2026 contest dataset on Modal, then benchmarks a copy that adds trivial pointer-based memoization. The memoized version returns exactly the same outputs, but it appears `12.90x` faster because the benchmark repeatedly calls the same workload tensors after a warmup pass.

That is the bug: memoization is not a kernel improvement. It exploits repeated identical benchmark inputs, not faster compute. A benchmark that treats this as a win can rank cache hits over real kernel work.

## Minimal Repro

1. Create a local venv and install Modal:

```bash
python3 -m venv .venv
.venv/bin/pip install modal
```

2. Authenticate Modal once if needed:

```bash
.venv/bin/modal setup
```

3. Run the Modal benchmark on B200:

```bash
.venv/bin/modal run scripts/benchmark_modal.py
```

## Outputs

- Raw Modal JSON: `benchmark_artifacts/modal_results.json`
- Baseline latency text: `benchmark_solutions/gdn_prefill_example/latency.txt`
- Memoized latency text: `benchmark_solutions/gdn_prefill_example_memoized/latency.txt`

## Benchmark Configuration

- Warmup runs: `1`
- Iterations: `5`
- Trials: `3`

## Comparison Results

| Workload UUID | Baseline Mean (ms) | Memoized Mean (ms) | Apparent Speedup |
|---|---:|---:|---:|
| `ef9515b6-ad88-4a3e-bd89-31384ddd53ad` | 0.1705 | 0.0127 | 13.41x |
| `9343fd82-a06d-493a-9918-1044b0c1cbd1` | 0.1443 | 0.0121 | 11.88x |
| `87bff084-1b7a-478e-99fd-5952c989d80e` | 0.1388 | 0.0116 | 12.01x |
| `3215fe5f-4a3b-4eb6-af20-4e17368d87a9` | 0.2074 | 0.0130 | 16.00x |
| `c5257f65-c411-4dbb-9dc1-ad4abcf00254` | 0.1904 | 0.0166 | 11.44x |

- Baseline average latency: `0.1703 ms`
- Memoized average latency: `0.0132 ms`
- Apparent average speedup from memoization: `12.90x`

## Correctness Check For Memoized Copy

| Workload UUID | Output Diff | State Diff |
|---|---:|---:|
| `ef9515b6-ad88-4a3e-bd89-31384ddd53ad` | `0` | `0` |
| `9343fd82-a06d-493a-9918-1044b0c1cbd1` | `0` | `0` |
| `c5257f65-c411-4dbb-9dc1-ad4abcf00254` | `0` | `0` |
| `87bff084-1b7a-478e-99fd-5952c989d80e` | `0` | `0` |
| `3215fe5f-4a3b-4eb6-af20-4e17368d87a9` | `0` | `0` |
