#!/usr/bin/env python3
"""
Benchmark the official gdn_prefill example solution and a memoized clone on Modal.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLUTIONS_ROOT = PROJECT_ROOT / "benchmark_solutions"
ARTIFACTS_ROOT = PROJECT_ROOT / "benchmark_artifacts"

DEFINITION_URL = (
    "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/resolve/main/"
    "definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json"
)
WORKLOADS_URL = (
    "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/resolve/main/"
    "workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl"
)
BLOB_URL_PREFIX = (
    "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/resolve/main/"
    "blob/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last"
)

SELECTED_WORKLOADS = [
    "ef9515b6-ad88-4a3e-bd89-31384ddd53ad",
    "9343fd82-a06d-493a-9918-1044b0c1cbd1",
    "87bff084-1b7a-478e-99fd-5952c989d80e",
    "3215fe5f-4a3b-4eb6-af20-4e17368d87a9",
    "c5257f65-c411-4dbb-9dc1-ad4abcf00254",
]

WARMUP_RUNS = 1
ITERATIONS = 5
NUM_TRIALS = 3

REMOTE_ASSETS_ROOT = "/root/bench_assets"
REMOTE_SOLUTIONS_ROOT = f"{REMOTE_ASSETS_ROOT}/benchmark_solutions"
REMOTE_DATA_ROOT = Path("/tmp/gdn_prefill_subset")

app = modal.App("flashinfer-bench-methodology-bug")
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .pip_install("safetensors")
    .add_local_dir(str(SOLUTIONS_ROOT), remote_path=REMOTE_SOLUTIONS_ROOT)
)


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


@app.function(image=image, gpu="B200:1", timeout=3600)
def run_remote_benchmark() -> dict:
    import importlib
    import json
    import sys
    import urllib.request
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    selected = set(SELECTED_WORKLOADS)
    data_root = REMOTE_DATA_ROOT
    data_root.mkdir(parents=True, exist_ok=True)
    blobs_root = data_root / "blob"
    blobs_root.mkdir(parents=True, exist_ok=True)

    definition = json.loads(_read_url(DEFINITION_URL).decode("utf-8"))
    workload_rows = [
        json.loads(line)
        for line in _read_url(WORKLOADS_URL).decode("utf-8").splitlines()
        if line.strip()
    ]
    workload_rows = [
        row for row in workload_rows if row["workload"]["uuid"] in selected
    ]

    for row in workload_rows:
        workload = row["workload"]
        sample_name = Path(workload["inputs"]["q"]["path"]).name
        sample_path = blobs_root / sample_name
        if not sample_path.exists():
            sample_url = f"{BLOB_URL_PREFIX}/{sample_name}"
            with urllib.request.urlopen(sample_url) as response:
                sample_path.write_bytes(response.read())
        workload["local_blob_path"] = str(sample_path)

    sys.path.insert(0, REMOTE_SOLUTIONS_ROOT)
    importlib.invalidate_caches()

    def load_solution_module(solution_name: str):
        package_name = f"{solution_name}.main"
        if package_name in sys.modules:
            del sys.modules[package_name]
        if solution_name in sys.modules:
            del sys.modules[solution_name]
        importlib.invalidate_caches()
        return importlib.import_module(package_name)

    def load_inputs(workload: dict) -> dict:
        tensors = load_file(workload["local_blob_path"])
        inputs = {}
        for input_name, input_meta in workload["inputs"].items():
            if input_meta["type"] == "scalar":
                inputs[input_name] = float(input_meta["value"])
            else:
                tensor = tensors[input_meta["tensor_key"]]
                inputs[input_name] = tensor.cuda()
        return inputs

    def benchmark_solution(solution_name: str) -> list[dict]:
        module = load_solution_module(solution_name)
        rows = []
        for row in workload_rows:
            workload = row["workload"]
            inputs = load_inputs(workload)

            if hasattr(module, "clear_cache"):
                module.clear_cache()

            baseline_output = None
            baseline_state = None
            if solution_name.endswith("_memoized"):
                base_module = load_solution_module("gdn_prefill_example")
                baseline_output, baseline_state = base_module.run(**inputs)
                if hasattr(module, "clear_cache"):
                    module.clear_cache()

            for _ in range(WARMUP_RUNS):
                module.run(**inputs)
            torch.cuda.synchronize()

            trial_latencies = []
            for _ in range(NUM_TRIALS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(ITERATIONS):
                    output, new_state = module.run(**inputs)
                end.record()
                torch.cuda.synchronize()
                trial_latencies.append(start.elapsed_time(end) / ITERATIONS)

            row_result = {
                "uuid": workload["uuid"],
                "total_seq_len": workload["axes"]["total_seq_len"],
                "num_seqs": workload["axes"]["num_seqs"],
                "trial_latencies_ms": trial_latencies,
                "mean_latency_ms": sum(trial_latencies) / len(trial_latencies),
            }

            if baseline_output is not None:
                row_result["max_abs_diff_output"] = float(
                    (output.float() - baseline_output.float()).abs().max().item()
                )
                row_result["max_abs_diff_state"] = float(
                    (new_state.float() - baseline_state.float()).abs().max().item()
                )

            rows.append(row_result)
        return rows

    baseline_rows = benchmark_solution("gdn_prefill_example")
    memoized_rows = benchmark_solution("gdn_prefill_example_memoized")

    return {
        "definition": definition["name"],
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "iterations": ITERATIONS,
            "num_trials": NUM_TRIALS,
        },
        "solutions": {
            "gdn_prefill_example": baseline_rows,
            "gdn_prefill_example_memoized": memoized_rows,
        },
    }


def write_latency_text(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        trial_text = ", ".join(f"{value:.4f}" for value in row["trial_latencies_ms"])
        lines.append(
            f"{row['uuid']} total_seq_len={row['total_seq_len']} "
            f"num_seqs={row['num_seqs']} mean_latency_ms={row['mean_latency_ms']:.4f} "
            f"trials_ms=[{trial_text}]"
        )
    path.write_text("\n".join(lines) + "\n")


@app.local_entrypoint()
def main() -> None:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not (SOLUTIONS_ROOT / "gdn_prefill_example").exists():
        raise SystemExit(
            "Missing benchmark solution folders under benchmark_solutions/."
        )

    results = run_remote_benchmark.remote()
    baseline_rows = results["solutions"]["gdn_prefill_example"]
    memoized_rows = results["solutions"]["gdn_prefill_example_memoized"]

    base_dir = SOLUTIONS_ROOT / "gdn_prefill_example"
    memoized_dir = SOLUTIONS_ROOT / "gdn_prefill_example_memoized"

    write_latency_text(base_dir / "latency.txt", baseline_rows)
    write_latency_text(memoized_dir / "latency.txt", memoized_rows)
    (ARTIFACTS_ROOT / "modal_results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {base_dir / 'latency.txt'}")
    print(f"Wrote {memoized_dir / 'latency.txt'}")
    print(f"Wrote {ARTIFACTS_ROOT / 'modal_results.json'}")
