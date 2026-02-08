#include <iostream>
#include <cmath>

using namespace std;

void initializeMatrix(float* a, int M, int N){
    for(int i{0}; i < M; i++) {
        for (int j{0}; j < N; j++){
            a[i * N + j] = i * N + j;
        }
    }
    return;
}

void printMatrix(float* a, int M, int N){
    for(int i{0}; i < M; i++) {
        for(int j{0}; j < N; j++) cout << a[i * N + j] << ' ';
        cout << '\n';
    }
    return;
}

__global__ void conv2dKernel(float* input, float* output, float* kernel, int M, int N, int K){
    // Compute the index of current thread in the launched cuda grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Map the current thread to compute
    // Here current thread (row, col) would compute the element (row, col) of the output image

    int Mout{M - K + 1}, Nout{N - K + 1};
    if(row < Mout && col < Nout){
        // Iterate over the kernel and compute the dot product with the overlapping image region
        float dotProduct{0.0f};
        for(int ki{0}; ki < K; ki++){
            for(int kj{0}; kj < K; kj++){
                dotProduct += kernel[ki * K + kj] * input[(row + ki) * N + (col + kj)];
            }
        }
        output[row * Nout + col] = dotProduct;
    }
    return;
}

void conv2d(){
    // Input Shape: M x N, Kernel Shape: K x K, Output Shape: M-K+1 x N-K+1
    int M{4}, N{5}, K{2};
    int MOut{M - K + 1}, NOut{N - K + 1};

    // Allocate host memory for input, output images and convolution filter
    float* in = (float*) malloc(sizeof(float) * M * N);
    float* out = (float*) malloc(sizeof(float) * MOut * NOut);
    float* kernel = (float*) malloc(sizeof(float) * K * K);

    // Initialize the input image and filter
    initializeMatrix(in, M, N);
    initializeMatrix(kernel, K, K);

    // Allocate device memory
    float *din, *dout, *dkernel;
    cudaMalloc(&din, M * N * sizeof(float));
    cudaMalloc(&dout, MOut * NOut * sizeof(float));
    cudaMalloc(&dkernel, K * K * sizeof(float));

    // Initialize device memory
    cudaMemcpy(din, in, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkernel, kernel, K * K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch convolution 2D kernel
    // Here we launch a 2D cuda grid equal to the shape of the output image
    dim3 threadsPerBlock{16, 16, 1};
    dim3 blocksPerGrid{(NOut + threadsPerBlock.x - 1) / threadsPerBlock.x, (MOut + threadsPerBlock.y - 1) / threadsPerBlock.y, 1};
    conv2dKernel<<<blocksPerGrid, threadsPerBlock>>>(din, dout, dkernel, M, N, K);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(out, dout, MOut * NOut * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the inputs and results
    printMatrix(in, M, N);
    printMatrix(kernel, K, K);
    printMatrix(out, MOut, NOut);

    // Free allocated host memory
    cudaFree(din); cudaFree(dout), cudaFree(dkernel);
    free(in); free(out); free(kernel);

    return;
}

int main(){
    conv2d();
}