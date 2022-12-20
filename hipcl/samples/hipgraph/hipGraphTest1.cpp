#include <iostream>
#include <cmath>
#include <vector>

// hip header file
#include "hip/hip_runtime.h"


#define WIDTH 64

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    __shared__ float sharedMem[WIDTH * WIDTH];

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    int i = 0;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    hipStream_t streamForGraph;
    hipStreamCreate(&streamForGraph);

    // Create graph
    hipGraph_t graph;
    hipGraphCreate(&graph, 0);

    std::vector<hipGraphNode_t> nodeDependencies;
    hipGraphNode_t kernelNode;
    hipKernelNodeParams kernelNodeParams = { 0 };

    kernelNodeParams.func = (void*)matrixTranspose;
    kernelNodeParams.gridDim = dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    kernelNodeParams.sharedMemBytes = 0;
    int width = WIDTH;
    void* kernelArgs[3] = {(void*)&gpuTransposeMatrix, (void*)&gpuMatrix, &width};
    // { (void*)gpuTransposeMatrix, (void*)gpuMatrix, &width};
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = NULL;

    // Create graph node
    hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

    // Create graph execution
    hipGraphExec_t graphExec;
    hipGraphInstantiate(&graphExec, graph, 0);

    // Launch graph
     hipGraphLaunch(graphExec, streamForGraph);
    // Synchronize stream
    hipStreamSynchronize(streamForGraph);

    // Lauching kernel from host
    // hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //                 dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
    //                 gpuMatrix, WIDTH);

    
    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    float eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            printf("%d cpu: %f gpu  %f\n", i, cpuTransposeMatrix[i], TransposeMatrix[i]);
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
