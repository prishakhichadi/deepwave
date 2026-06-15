"""Generates a synthetic rocprof profiling corpus covering all 4 bottleneck types.
Each scenario produces a realistic CSV + a paired HIP kernel with a known bottleneck.
Used for eval/testing when real hardware profiling data is unavailable."""

import csv
import os
from pathlib import Path

OUTPUT_DIR = Path("corpus/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic metric profiles — values grounded in real AMD MI300X behavior
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "memory_bandwidth_bound",
        "description": "Strided global memory access, no shared memory — classic BW bottleneck",
        "metrics": {
            "VALUUtilization":  18.3,
            "SALUUtilization":   8.1,
            "MemUnitStalled":   72.4,
            "MaxWavesPerCU":    24.0,
            "L2CacheHit":       31.2,
            "LDSBankConflict":   0.0,
        },
        "kernel": """\
#include <hip/hip_runtime.h>

// BOTTLENECK: Strided memory access — threads access non-consecutive addresses.
// Each thread reads A[i * stride] causing cache line waste on MI300X.
__global__ void strided_copy(float* __restrict__ out, const float* __restrict__ in,
                              int stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i * stride];  // Strided — uncoalesced global load
    }
}
"""
    },
    {
        "name": "compute_bound",
        "description": "Dense FP32 multiply-accumulate loop — VALU pipes saturated",
        "metrics": {
            "VALUUtilization":  91.7,
            "SALUUtilization":  12.3,
            "MemUnitStalled":   11.2,
            "MaxWavesPerCU":    28.0,
            "L2CacheHit":       82.1,
            "LDSBankConflict":   0.0,
        },
        "kernel": """\
#include <hip/hip_runtime.h>

// BOTTLENECK: Compute bound — tight inner loop with no memory pressure.
// VALU pipes are fully saturated. Candidate for FMA fusion or reduced precision.
__global__ void dot_product_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (i < n) {
        for (int k = 0; k < 128; k++) {
            sum += a[i * 128 + k] * b[i * 128 + k];  // Dense MAD loop
        }
        out[i] = sum;
    }
}
"""
    },
    {
        "name": "occupancy_limited",
        "description": "High register usage + large shared memory block limits waves per CU",
        "metrics": {
            "VALUUtilization":  34.1,
            "SALUUtilization":  14.2,
            "MemUnitStalled":   28.3,
            "MaxWavesPerCU":     8.0,   # Well below 16 threshold
            "L2CacheHit":       61.0,
            "LDSBankConflict":   2.1,
        },
        "kernel": """\
#include <hip/hip_runtime.h>

// BOTTLENECK: Occupancy limited — large shared memory allocation (48KB/block)
// restricts the number of concurrent wavefronts per CU on MI300X.
__global__ void matrix_transpose(float* out, const float* in, int width, int height) {
    __shared__ float tile[32][33];  // 32x33 to avoid bank conflicts, but 4KB per block

    // Many local variables — high register pressure
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int r0, r1, r2, r3, r4, r5, r6, r7;  // Excess register usage
    r0 = x; r1 = y; r2 = x+1; r3 = y+1; r4 = x+2; r5 = y+2; r6 = x+3; r7 = y+3;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < height && y < width)
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];

    (void)(r0+r1+r2+r3+r4+r5+r6+r7);  // Prevent optimizer from eliminating registers
}
"""
    },
    {
        "name": "latency_bound",
        "description": "Low utilization across all units — pipeline stalls from dependent loads",
        "metrics": {
            "VALUUtilization":  22.1,
            "SALUUtilization":   9.4,
            "MemUnitStalled":   19.8,
            "MaxWavesPerCU":    12.0,
            "L2CacheHit":       55.3,
            "LDSBankConflict":   0.5,
        },
        "kernel": """\
#include <hip/hip_runtime.h>

// BOTTLENECK: Latency bound — pointer-chasing load pattern creates long dependency chains.
// Each load depends on the result of the previous one, stalling the pipeline.
__global__ void pointer_chase(int* out, const int* indices, const float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = indices[i];           // Load 1
        idx = indices[idx];             // Load 2 — depends on load 1 (latency chain)
        idx = indices[idx];             // Load 3 — depends on load 2
        out[i] = (int)data[idx];        // Load 4 — depends on load 3
    }
}
"""
    },
]


def generate_csv(metrics: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"  CSV  → {path}")


def generate_kernel(kernel_code: str, path: Path) -> None:
    path.write_text(kernel_code, encoding="utf-8")
    print(f"  HIP  → {path}")


def main():
    print("Generating synthetic profiling corpus...\n")
    for scenario in SCENARIOS:
        name = scenario["name"]
        scenario_dir = OUTPUT_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{name}] {scenario['description']}")
        generate_csv(scenario["metrics"], scenario_dir / "rocprof.csv")
        generate_kernel(scenario["kernel"],  scenario_dir / "kernel.hip")
        print()

    print(f"Done. {len(SCENARIOS)} scenarios written to {OUTPUT_DIR}/")
    print("\nUsage:")
    print("  python main.py --kernel corpus/synthetic/memory_bandwidth_bound/kernel.hip \\")
    print("                 --profiling corpus/synthetic/memory_bandwidth_bound/rocprof.csv")


if __name__ == "__main__":
    main()