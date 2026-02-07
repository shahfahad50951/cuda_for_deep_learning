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

void softmax_kernel(float* input, float* output, int M, int N){
    // Outer loop: Processes one row at a time
    for(int row{0}; row < M; row++){
        // Compute max for the row
        float max_val{-1e20f};
        for(int col{0}; col < N; col++){
            max_val = max(max_val, input[row * N + col]);
        }

        // Compute sum of exponentials, max subtracted
        float sum{0.0f};
        for(int col{0}; col < N; col++){
            sum += expf(input[row * N + col] - max_val);
        }

        // Write out all the outputs
        for(int col{0}; col < N; col++){
            output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
        }
    }
    return;
}

void softmax(){
    int M{4}, N{5};
    float* in = (float*) malloc(sizeof(float) * M * N);
    float* out = (float*) malloc(sizeof(float) * M * N);

    initializeMatrix(in, M, N);

    softmax_kernel(in, out, M, N);

    printMatrix(in, M, N);
    printMatrix(out, M, N);
    return;
}

int main(){
    softmax();
}