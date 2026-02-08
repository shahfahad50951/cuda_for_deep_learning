#include <iostream>
#include <cmath>

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

__global__ void rmsnormKernel(float* input, float* output, int M, int N){
    // Compute the index of current thread in the launched 2D cuda grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Map current thread (row,col) to work
    // Here each thread would comptue the RMS normlized value of input[row][col] and write to output[row][col]
    if(row < M && col < N){
        float squaredSum{0.0f};
        for(int j{0}; j < N; j++){
            squaredSum += input[row * N + j] * input[row * N + j];
        }
        float rootMeanSquaredSum{sqrtf(squaredSum / N)};
        output[row * N + col] = input[row * N + col] / rootMeanSquaredSum;
    }
    return;
}

void rmsnorm(){
    int M{4}, N{5};

    float *in, *out;
    in = (float*) malloc(M * N * sizeof(float));
    out = (float*) malloc(M * N * sizeof(float));

    initializeMatrix(in, M, N);

    float *ind, *oud;
    cudaMalloc(&ind, M * N * sizeof(float));
    cudaMalloc(&oud, M * N * sizeof(float));

    cudaMemcpy(ind, in, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup the cuda launch grid and launch kernel
    dim3 threadsPerBlock{16, 16, 1};
    dim3 blocksPerGrid{(N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y, 1};
    rmsnormKernel<<<blocksPerGrid, threadsPerBlock>>>(ind, oud, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(out, oud, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(in, M, N);
    printMatrix(out, M, N);

    cudaFree(ind); cudaFree(oud);
    free(in); free(out);

    return;
}

int main(){
    rmsnorm();
    return 0;
}