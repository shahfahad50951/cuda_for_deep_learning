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

void rmsnormKernel(float* input, float* output, int M, int N){
    for(int row{0}; row < M; row++){
        // Iterate over the elements of the row to compute the RMS value
        float squaredSum{0.0f};
        for(int col{0}; col < N; col++){
            squaredSum += input[row * N + col] * input[row * N + col];
        }
        float rootMeanSquaredSum{sqrtf(squaredSum / N)};

        // Iterate over the elements of the row to normalize each element by the RMS value
        for(int col{0}; col < N; col++) {
            output[row * N + col] = input[row * N + col] / rootMeanSquaredSum;
        }
    }
    return;
}

void rmsnorm(){
    int M{4}, N{5};
    float* in = (float*) malloc(sizeof(float) * M * N);
    float* out = (float*) malloc(sizeof(float) * M * N);

    initializeMatrix(in, M, N);

    rmsnormKernel(in, out, M, N);

    printMatrix(in, M, N);
    printMatrix(out, M, N);
    return;
}

int main(){
    rmsnorm();
}