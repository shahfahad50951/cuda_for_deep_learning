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

__global__ void gemmKernel(float* A, float* B, float* C, int M, int N, int K){
    // Find the index of the current thread in the thread grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Map this thread to work
    // Here, each thread (row, col) would compute the entry (row, col) of the output matrix C
    // i.e the dot product of ith row of A and jth column of B
    if(row < M && col < N){
        for(int k{0}; k < K; k++){
            // C[i][j] += A[i][k] * B[k][j]
            C[row * N + col] += A[row * K + k] * B[k * N + col];
        } 
    }
    return;
}

void gemm(){
    int M{4}, N{3}, K{4};

    // Allocate host memory
    float *ha, *hb, *hc;
    ha = (float*) malloc(M * K * sizeof(float));
    hb = (float*) malloc(K * N * sizeof(float));
    hc = (float*) malloc(M * N * sizeof(float));

    // Initialize host memory
    initializeMatrix(ha, M, K, 1.0);
    initializeMatrix(hb, K, N, 1.0);
    initializeMatrix(hc, M, N, 0.0);
    
    // Allocate device memory
    float *da, *db, *dc;
    cudaMalloc(&da, M * K * sizeof(float));
    cudaMalloc(&db, K * N * sizeof(float));
    cudaMalloc(&dc, M * N * sizeof(float));
    
    // Transfer data from host to device and initialize device memory
    cudaMemcpy(da, ha, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dc, 0, M * N * sizeof(float));

    // Compue the grid and block dimensions
    // Launch the kernel such that it grid covers the entire output matrix C
    dim3 threadsPerBlock{16, 16, 1};
    dim3 blocksPerGrid{(N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y, 1};
    gemmKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, M, N, K);

    cudaDeviceSynchronize();

    // Transfer output from device to host
    cudaMemcpy(hc, dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the inputs and result
    printMatrix(ha, M, K);
    printMatrix(hb, K, N);
    printMatrix(hc, M, N);

    // Free up the memory
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return;
}

int main(){
    gemm();
    return 0;
}