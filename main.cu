/* 
 * File:   main.cu
 * Author: Zosia Sobocinska
 *
 * Created on November 20, 2013, 7:17 PM
 */

#include <stdio.h>
#include "mt19937-64.c"
#include "utils.cu"

#include "kernel.cu"


int main(int argc, char** argv) {

    //////////////////////////////////////////////////////////////////////////
    // USAGE

    uint size;
    switch (argc) {
        case 2:
            size = atoi(argv[1]);
            break;
        case 3:
            size = atoi(argv[2]);
            break;
        default:
            printf("Usage: %s [dummy] <matrix_size N=width=height>\n", argv[0]);
            exit(64);
    }

    char time_msg[20];
    const char* testrun = getenv("TESTRUN");
    if (testrun != NULL && (*testrun) == 'y') {
        strcpy(time_msg, "%d");
    } else {
        strcpy(time_msg, "%fms\n");
    }

    //////////////////////////////////////////////////////////////////////////
    // INPUT
    float* A_host;
    mtrx::host::alloc_mem(&A_host, size);
    mtrx::host::fill(&A_host, size);

    float* B_host;
    mtrx::host::alloc_mem(&B_host, size);
    mtrx::host::fill(&B_host, size);

    float* A_dev;
    mtrx::dev::alloc_mem(&A_dev, size);

    float* B_dev;
    mtrx::dev::alloc_mem(&B_dev, size);

    //////////////////////////////////////////////////////////////////////////
    // OUTPUT

    float* C_host;
    mtrx::host::alloc_mem(&C_host, size);

    float* C_dev;
    mtrx::dev::alloc_mem(&C_dev, size);

    //////////////////////////////////////////////////////////////////////////
    // SEND TO GPU

    mtrx::host::cuda_host2dev(A_host, A_dev, size);
    mtrx::host::cuda_host2dev(B_host, B_dev, size);

    //////////////////////////////////////////////////////////////////////////
    // BLOCKS AND GRID DIMENSIONS
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE) / BLOCK_SIZE - 1, (size + BLOCK_SIZE) / BLOCK_SIZE - 1);

    //////////////////////////////////////////////////////////////////////////
    // TIME MEASUREMENT START
    float time = 1.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //////////////////////////////////////////////////////////////////////////
    // RUN KERNEL
    }

    //////////////////////////////////////////////////////////////////////
    // TIME MEASUREMENT STOP
    mtrx::dev::test(cudaEventRecord(stop, NULL)); // stop time measurement
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf(time_msg, time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //////////////////////////////////////////////////////////////////////
    // RETRIEVE FROM GPU
    mtrx::dev::cuda_dev2host(C_dev, C_host, size);
    // do something with calculated matrix... (printf maybe?)

    //////////////////////////////////////////////////////////////////////////
    // FREE MEMORY
    mtrx::host::free_mem(&A_host);
    mtrx::host::free_mem(&B_host);
    mtrx::host::free_mem(&C_host);

    mtrx::dev::free_mem(&A_dev);
    mtrx::dev::free_mem(&B_dev);
    mtrx::dev::free_mem(&C_dev);


}
