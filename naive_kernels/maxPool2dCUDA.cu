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

__global__ void maxPool2dKernel(float* input, float* output, int M, int N, int K){
    // Find the index of current thread in the launched cuda grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Map the current thread (row,col) to some work
    // Here current thread would be responsible for computing the output entry (row,col)
    int Mout{M / K}, Nout{N / K};
    if(row < Mout && col < Nout){
        float maxVal{-1e20f};
        for(int pi{0}; pi < K; pi++){
            for(int pj{0}; pj < K; pj++){
                int inputRow{row * K + pi};
                int inputCol{col * K + pj};
                maxVal = max(maxVal, input[inputRow * N + inputCol]);
            }
        }
        output[row * Nout + col] = maxVal;
    }
    return;
}

void maxPool2d(){
    // Input Shape: M x N, Pooling Window Shape: K x K, Output Shape: M/K x N/K
    int M{4}, N{5}, K{2};
    int MOut{M / K}, NOut{N / K};

    // Allocate host memory for input and output images
    float* in = (float*) malloc(sizeof(float) * M * N);
    float* out = (float*) malloc(sizeof(float) * MOut * NOut);

    // Initialize the input image
    initializeMatrix(in, M, N);

    // Allocate device memory
    float *din, *dout;
    cudaMalloc(&din, M * N * sizeof(float));
    cudaMalloc(&dout, MOut * NOut * sizeof(float));

    // Initialize device memory
    cudaMemcpy(din, in, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch maxPool 2D kernel
    // Here we launch a 2D cuda grid equal to the shape of the output image
    dim3 threadsPerBlock{16, 16, 1};
    dim3 blocksPerGrid{(NOut + threadsPerBlock.x - 1) / threadsPerBlock.x, (MOut + threadsPerBlock.y - 1) / threadsPerBlock.y, 1};
    maxPool2dKernel<<<blocksPerGrid, threadsPerBlock>>>(din, dout, M, N, K);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(out, dout, MOut * NOut * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the inputs and results
    printMatrix(in, M, N);
    printMatrix(out, MOut, NOut);

    // Free allocated host memory
    cudaFree(din); cudaFree(dout);
    free(in); free(out);

    return;
}

int main(){
    maxPool2d();
}