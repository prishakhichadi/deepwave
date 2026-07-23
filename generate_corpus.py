import csv
import os
from pathlib import Path

OUTPUT_DIR = Path("corpus/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



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
            "MaxWavesPerCU":     8.0,   
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
    {
        "name": "lds_bank_conflict_bound",
        "description": "Unpadded shared memory tile with strided access — LDS bank conflicts",
        "metrics": {
            "VALUUtilization":  42.6,
            "SALUUtilization":  10.5,
            "MemUnitStalled":   21.7,
            "MaxWavesPerCU":    22.0,
            "L2CacheHit":       68.4,
            "LDSBankConflict":  27.3,
            "VGPRCount":        38.0,
        },
        "kernel": """\
#include <hip/hip_runtime.h>

// BOTTLENECK: LDS bank conflict — unpadded shared tile, strided intra-block access.
// tile[row][col * STRIDE] lands multiple threads in a wavefront on the same one of
// the 32 LDS banks on MI300X, serializing what should be a single-cycle LDS read.
#define STRIDE 8
__global__ void banked_reduce(float* out, const float* in, int n) {
    __shared__ float tile[64];  // no padding — 64 == 2*32 banks, conflict-prone

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        tile[tid] = in[gid];
    }
    __syncthreads();

    // Strided shared-memory read — every STRIDE-th thread hits the same bank.
    float v = tile[(tid * STRIDE) % 64];
    if (gid < n) {
        out[gid] = v;
    }
}
"""
    },
    {
        "name": "register_pressure_bound",
        "description": "Excessive live scalars + a private local array — register spill risk",
        "metrics": {
            "VALUUtilization":  55.8,
            "SALUUtilization":  18.2,
            "MemUnitStalled":   24.1,
            "MaxWavesPerCU":    19.0,  
            "L2CacheHit":       71.5,
            "LDSBankConflict":   1.2,
            "VGPRCount":        96.0,   
        },
        "kernel": """\
#include <hip/hip_runtime.h>

// BOTTLENECK: Register pressure — many live scalars plus a private per-thread array
// that the compiler can't fully keep in registers, forcing spills to local memory.
__global__ void heavy_stencil(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Large private array per thread — classic spill trigger, not shared across threads.
    float window[16];
    for (int k = 0; k < 16; k++) {
        int idx = i + k - 8;
        window[k] = (idx >= 0 && idx < n) ? in[idx] : 0.0f;
    }

    // Many simultaneously-live scalars compound the pressure.
    float a0=window[0], a1=window[1], a2=window[2], a3=window[3];
    float a4=window[4], a5=window[5], a6=window[6], a7=window[7];
    float a8=window[8], a9=window[9], a10=window[10], a11=window[11];
    float a12=window[12], a13=window[13], a14=window[14], a15=window[15];

    out[i] = (a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15) / 16.0f;
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