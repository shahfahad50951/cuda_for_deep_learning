#include <iostream>
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

void gemmKernel(float* A, float* B, float* C, int M, int N, int K){
    // Loop across all the output entries and for each entry (i, j) compute the dot product of
    // ith row vector of A, and jth column vector of B
    for(int i{0}; i < M; i++){
        for(int j{0}; j < N; j++){
            for(int k{0}; k < K; k++){
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    return;
}

void gemm(){
    int M{4}, N{3}, K{4};

    float *ha, *hb, *hc;
    ha = (float*) malloc(M * K * sizeof(float));
    hb = (float*) malloc(K * N * sizeof(float));
    hc = (float*) malloc(M * N * sizeof(float));

    initializeMatrix(ha, M, K, 1.0);
    initializeMatrix(hb, K, N, 1.0);
    initializeMatrix(hc, M, N, 0.0);
    gemmKernel(ha, hb, hc, M, N, K);
    
    printMatrix(ha, M, K);
    printMatrix(hb, K, N);
    printMatrix(hc, M, N);
    
    free(ha); free(hb); free(hc);
    return;
}

int main(){
    gemm();
    return 0;
}