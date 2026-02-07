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

void matrixTransposeKernel(float* in, int M, int N, float* out){
    for(int i{0}; i < M; i++){
        for(int j{0}; j < N; j++){
            out[j * M + i] = in[i * N + j];
        }
    }
    return;
}

void matrixTranspose(){
    int M{4}, N{6};

    float *hi, *ho;
    hi = (float*) malloc(M * N * sizeof(float));
    ho = (float*) malloc(M * N * sizeof(float));

    initializeMatrix(hi, M, N, 1.0);
    matrixTransposeKernel(hi, M, N, ho);
    
    printMatrix(hi, M, N);
    printMatrix(ho, N, M);
    
    free(hi); free(ho);
    return;
}

int main(){
    matrixTranspose();
    return 0;
}