/* 
 * File:   main.cu
 * Author: Zosia Sobocinska
 *
 * Created on November 20, 2013, 7:17 PM
 */

#include <stdio.h>
#include "mt19937-64.c"
#include "utils.h"

#include "kernel.cu"
#define BLOCK_SIZE 4


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

    // dimensions of the blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // dimensions of the grid
    dim3 grid(size / threads.x, size / threads.y);

    //////////////////////////////////////////////////////////////////////////
    // CREATE EVENTS FOR TIME MEASUREMENT
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //////////////////////////////////////////////////////////////////////////
    // RUN KERNEL
    cudaEventRecord(start, 0); // start time measurement
    multiple << < grid, threads >> >(C_dev, A_dev, B_dev, size); // run
    cudaEventRecord(stop, 0); // stop time measurement
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    //////////////////////////////////////////////////////////////////////////
    // CALC EXECUTION TIME
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf(time_msg, time);

    //////////////////////////////////////////////////////////////////////////
    // RETRIEVE FROM GPU

    mtrx::dev::cuda_dev2host(C_dev, C_host, size);

    //////////////////////////////////////////////////////////////////////////
    // FREE MEMORY

    mtrx::host::free_mem(&A_host);
    mtrx::host::free_mem(&B_host);
    mtrx::host::free_mem(&C_host);

    mtrx::dev::free_mem(&A_dev);
    mtrx::dev::free_mem(&B_dev);
    mtrx::dev::free_mem(&C_dev);


}
