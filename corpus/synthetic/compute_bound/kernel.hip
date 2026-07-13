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
