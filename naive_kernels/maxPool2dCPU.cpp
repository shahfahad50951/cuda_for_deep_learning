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

void maxPool2dKernel(float* input, float* output, int M, int N, int K){
    int Mout{M / K}, Nout{N / K};
    
    // Loop over all output elements
    for(int i{0}; i < Mout; i++){
        for(int j{0}; j < Nout; j++){
            // For each output element, find the area of the input image that this output element represents
            // Take the minimum value of elements from that area
            float maxVal{-1e20f};
            for(int pi{0}; pi < K; pi++){
                for(int pj{0}; pj < K; pj++){
                    int inputRow{i * K + pi};
                    int inputCol{j * K + pj};
                    maxVal = max(maxVal, input[inputRow * N + inputCol]);
                }
            }
            output[i * Nout + j] = maxVal;
        }
    }
    return;
}

void maxPool2d(){
    // Input Shape: M x N, Pooling Window Shape: K x K
    // We will assume non-overlapping pooling with stride equal to the window length
    // If the dimension of input is not divisible by window length, we do floor division and discard the
    // pixels towards the last rows and columns
    int M{4}, N{5}, K{2};
    int MOut{M / K}, NOut{N / K};

    // Allocate host memory for input, output images
    float* in = (float*) malloc(sizeof(float) * M * N);
    float* out = (float*) malloc(sizeof(float) * MOut * NOut);

    // Initialize the input image
    initializeMatrix(in, M, N);

    // Launch maxPooling 2D kernel
    maxPool2dKernel(in, out, M, N, K);

    // Print the inputs and results
    printMatrix(in, M, N);
    printMatrix(out, MOut, NOut);

    // Free allocated host memory
    free(in); free(out);

    return;
}

int main(){
    maxPool2d();
}