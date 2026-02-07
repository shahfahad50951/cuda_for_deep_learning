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

__global__ void softmaxKernel(float* input, float* output, int M, int N){
    // Compute the location of current thread in the cuda launch grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Assign current thread (row, col) to compute the softmax for one element of (row, col)
    // 1st Pass: Compute the max of the row
    // 2nd Pass: Compute the sum of exponentials of elements of the row (after subtracting max from each element of the row)
    // Output: Compute exponential of element assigned to this thread (max subtracted) and normalize by the sum of 
    //         exponentials (max subtracted) computed in the 2nd pass and write the result

    if (row < M && col < N){
        // 1st Pass
        float max_val{-1e20f}; // Set max to a very large negative value (sentinal value)
        for(int i{0}; i < N; i++){
            if(input[row * N + i] > max_val) max_val = input[row * N + i];
        }

        // 2nd Pass
        float sum{0.0f};
        for(int i{0}; i < N; i++){
            sum += expf(input[row * N + i] - max_val);
        }

        // Output
        output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
    }
    return;
}

void softmax(){
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
    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(ind, oud, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(out, oud, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(in, M, N);
    printMatrix(out, M, N);

    cudaFree(ind); cudaFree(oud);
    free(in); free(out);

    return;
}

int main(){
    softmax();
    return 0;
}