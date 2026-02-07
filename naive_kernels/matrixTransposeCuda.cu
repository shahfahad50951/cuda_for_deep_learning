#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void initializeMatrix(float* m, int num_rows, int num_cols, float scale = 1.0){
    for(int row{0}; row < num_rows; row++){
        for(int col{0}; col < num_cols; col++){
            m[row * num_cols + col] = scale * (row * num_cols + col);
        }
    }
    return;
}

void printMatrix(float* m, int num_rows, int num_cols){
    for(int row{0}; row < num_rows; row++){
        for(int col{0}; col < num_cols; col++){
            cout << m[row * num_cols + col] << ' ';
        }
        cout << '\n';
    }
    return;
}

__global__ void matrixTransposeKernel(float* in, int M, int N, float* out){
    // Find unique identify of thread in the grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Now map the thread problem
    // Here we map this thread (row,col) to perform the transpose of (row, col) element of the input matrix
    if(row < M && col < N){
        out[col * M + row] = in[row * N + col];   
    }
    return;
}

void matrixTranspose(){
    int M{4}, N{6};

    float *hi, *ho;
    hi = (float*) malloc(M * N * sizeof(float));
    ho = (float*) malloc(M * N * sizeof(float));

    initializeMatrix(hi, M, N, 1.0);
    
    float *devi, *devo;
    cudaMalloc(&devi, M * N * sizeof(float));
    cudaMalloc(&devo, M * N * sizeof(float));
    
    cudaMemcpy(devi, hi, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Compue the grid and block dimensions
    // Launch the kernel
    dim3 threadsPerBlock{16, 16};
    dim3 blocksPerGrid{(N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y};
    matrixTransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(devi, M, N, devo);
    cudaDeviceSynchronize();

    cudaMemcpy(ho, devo, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(hi, M, N);
    printMatrix(ho, N, M);

    cudaFree(devi); cudaFree(devo);
    free(hi); free(ho);
    return;
}

int main(){
    matrixTranspose();
    return 0;
}