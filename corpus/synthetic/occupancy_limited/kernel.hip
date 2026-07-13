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
