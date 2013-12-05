/* 
 * File:   utils.h
 * Author: politechnika
 *
 * Created on December 5, 2013, 12:38 PM
 */

#ifndef UTILS_H
#define	UTILS_H

#include <stdio.h>

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

#endif	/* UTILS_H */

