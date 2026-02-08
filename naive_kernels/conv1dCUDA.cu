#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void initializeArray(float* a, int size){
    for(int i{0}; i < size; i++) a[i] = i;
    return;
}

void printArray(float* a, int size){
    for(int i{0}; i < size; i++) cout << a[i] << ' ';
    cout << '\n';
    return;
}

__global__ void conv1dKernel(float* input, float* output, float* kernel, int inputSize, int kernelSize){
    // Find the index of the current thread in the launched 1D cuda grid 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int outputSize{inputSize - kernelSize + 1};

    // Mapping from thread to work: This thread (idx) would be responsible for computing the element output[idx]
    if(idx < outputSize){
        float dotProduct{0.0f};
        for(int k{0}; k < kernelSize; k++){
            dotProduct += input[idx + k] * kernel[k];
        }
        output[idx] = dotProduct;
    }

    return;
}

void conv1d(){
    int inputSize{20}, kernelSize{3};
    int outputSize{inputSize - kernelSize + 1};

    float* in = (float*) malloc(sizeof(float) * inputSize);
    float* out = (float*) malloc(sizeof(float) * outputSize);
    float* kernel = (float*) malloc(sizeof(float) * kernelSize);

    initializeArray(in, inputSize);
    initializeArray(kernel, kernelSize);

    float *ind, *outd, *kerneld;
    cudaMalloc(&ind, inputSize * sizeof(float));
    cudaMalloc(&outd, outputSize * sizeof(float));
    cudaMalloc(&kerneld, kernelSize * sizeof(float));

    cudaMemcpy(ind, in, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kerneld, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Setup cuda launch grid and launch kernel
    dim3 threadsPerBlock{32, 1, 1};
    dim3 blocksPerGrid{(outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x};
    conv1dKernel<<<blocksPerGrid, threadsPerBlock>>>(ind, outd, kerneld, inputSize, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(out, outd, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    printArray(in, inputSize);
    printArray(kernel, kernelSize);
    printArray(out, outputSize);

    cudaFree(ind); cudaFree(outd); cudaFree(kerneld);
    free(in); free(out); free(kernel);

    return;
}

int main(){
    conv1d();
}