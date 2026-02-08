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

void conv2dKernel(float* input, float* output, float* kernel, int M, int N, int K){
    int Mout{M - K + 1}, Nout{N - K +1};
    for(int i{0}; i < Mout; i++){
        for(int j{0}; j < Nout; j++){
            float dotProduct{0.0f};
            for(int ki{0}; ki < K; ki++){
                for(int kj{0}; kj < K; kj++){
                    dotProduct += kernel[ki * K + kj] * input[(i + ki) * N + (j + kj)];
                }
            }
            output[i * Nout + j] = dotProduct;
        }
    }
    return;
}

void conv2d(){
    // Input Shape: M x N, Kernel Shape: K x K
    int M{4}, N{5}, K{2};
    int MOut{M - K + 1}, NOut{N - K + 1};

    // Allocate host memory for input, output images and convolution filter
    float* in = (float*) malloc(sizeof(float) * M * N);
    float* out = (float*) malloc(sizeof(float) * MOut * NOut);
    float* kernel = (float*) malloc(sizeof(float) * K * K);

    // Initialize the input image and filter
    initializeMatrix(in, M, N);
    initializeMatrix(kernel, K, K);

    // Launch convolution 2D kernel
    conv2dKernel(in, out, kernel, M, N, K);

    // Print the inputs and results
    printMatrix(in, M, N);
    printMatrix(kernel, K, K);
    printMatrix(out, MOut, NOut);

    // Free allocated host memory
    free(in); free(out); free(kernel);

    return;
}

int main(){
    conv2d();
}