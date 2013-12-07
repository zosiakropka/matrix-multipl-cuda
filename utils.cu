/* 
 * File:   utils.h
 * Author: politechnika
 *
 * Created on December 3, 2013, 12:38 PM
 */

#ifndef UTILS_CU
#define	UTILS_CU

#define BLOCK_SIZE 4

typedef unsigned int uint;
#include <stdio.h>

namespace mtrx {
    namespace host {
        void alloc_mem(float** matrix, uint size);
        void free_mem(float** matrix);
        void fill(float** matrix, uint size);
        void cuda_host2dev(float *host_matrix, float *dev_matrix, uint size);
    }
    namespace dev {
        void alloc_mem(float** matrix, uint size);
        void free_mem(float** matrix);
        void cuda_dev2host(float* dev_matrix, float* host_matrix, uint size);
        void test(cudaError_t err);
    }

    //////////////////////////////////////////////////////////////////////////
    namespace host {

        void alloc_mem(float** matrix, uint size) {
            (*matrix) = (float*) malloc(size * size * sizeof (float));
            memset((*matrix), 0, size * size * sizeof (float));
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
            cudaMemcpy(dev_matrix, host_matrix, size * size * sizeof (float), cudaMemcpyHostToDevice);
        }

    }
    namespace dev {

        void alloc_mem(float** matrix, uint size) {
            test(cudaMalloc(matrix, size * size * sizeof (float)));
        }

        void free_mem(float** matrix) {
            test(cudaFree((*matrix)));
            (*matrix) = NULL;
        }

        void cuda_dev2host(float* dev_matrix, float* host_matrix, uint size) {
            test(cudaMemcpy(host_matrix, dev_matrix, size * size * sizeof (float), cudaMemcpyDeviceToHost));
        }

        void test(cudaError_t err) {
            if (err != cudaSuccess) {
                fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
        }
    }
}

#endif	/* UTILS_CU */

