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
