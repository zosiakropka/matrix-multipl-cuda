
__global__ void multiple(float* C, float* A, float* B, uint size) {

    float c = 0.0;

    uint yc = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    uint xc = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    uint GRID_SIZE = (BLOCK_SIZE + size - 1) / BLOCK_SIZE;

    if (xc < size && yc < size) {

        for (uint i = 0; i < GRID_SIZE; i++) {
            for (uint j = 0; j < BLOCK_SIZE; ++j) {
                uint index = i * BLOCK_SIZE + j;
                if ((index < size) && (index < size)) {
                    c += A[yc * size + index] * B[index * size + xc];
                }
            }
        }
        C[yc * size + xc] = c;
    }
}
