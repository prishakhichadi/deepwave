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
