/* 
 * File:   main.cu
 * Author: Zosia Sobocinska
 *
 * Created on November 20, 2013, 7:17 PM
 */


#define BLOCK_SIZE 4

__global__ void multiple(float* C, float* A, float* B, uint size) {
}

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
