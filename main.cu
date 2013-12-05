/* 
 * File:   main.cu
 * Author: Zosia Sobocinska
 *
 * Created on November 20, 2013, 7:17 PM
 */

#include <stdio.h>
#include "mt19937-64.c"

#define BLOCK_SIZE 4

__global__ void multiple(float* C, float* A, float* B, uint size) {

    int x = threadIdx.x;
    int y = threadIdx.y;

    float C_yx = 0;
    for (uint i = 0; i < size; ++i) {
        float A_yi = A[y * size + i];
        float B_ix = B[i * size + x];
        C_yx += A_yi * B_ix;
    }

    C[y * size + x] = C_yx;
}

namespace mtrx {
    namespace host {

        void alloc_mem(float** matrix, uint size) {
            (*matrix) = (float*) malloc(size * size * sizeof (float));
        }

        void free_mem(float** matrix) {
            free((*matrix));
            (*matrix) = NULL;
        }

        void fill(float** matrix, uint size) {
            for (uint i = 0; i < size * size; ++i)
                (*matrix)[i] = (float) genrand64_real1();
        }

        void cuda_host2dev(float *host_matrix, float *dev_matrix, uint size) {
            cudaMemcpy(dev_matrix, host_matrix, size*size, cudaMemcpyHostToDevice);
        }

    }
    namespace dev {

        void alloc_mem(float** matrix, uint size) {
            cudaMalloc((void**) matrix, size * size * sizeof (float));
        }

        void free_mem(float** matrix) {
            cudaFree((*matrix));
            (*matrix) = NULL;
        }

        void cuda_dev2host(float* dev_matrix, float* host_matrix, uint size) {
            cudaMemcpy(host_matrix, dev_matrix, size*size, cudaMemcpyDeviceToHost);
        }
    }
}

/**
 */
int main(int argc, char** argv) {

    //////////////////////////////////////////////////////////////////////////
    // USAGE

    if (argc != 3) {
        printf("Usage: %s <dummy> <matrix_size N=width=height>", argv[0]);
    }
    char time_msg[20];
    const char* testrun = getenv("TESTRUN");
    if (testrun != NULL && (*testrun) == 'y') {
        strcpy(time_msg, "%d");
    } else {
        strcpy(time_msg, "%dms\n");
    }

    uint size = atoi(argv[2]);

}
