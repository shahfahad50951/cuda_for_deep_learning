#include <iostream>

using namespace std;

__global__ void cudaAdd2DKernel(float* a, float* b, float* c, int num_rows, int num_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows && col < num_cols){
        int linear_idx = row * num_cols + col;
        c[linear_idx] = a[linear_idx] + b[linear_idx];
    }
    return;
}

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

void cudaAdd2D(){
    // int num_rows{1000}, num_cols{500};
    int num_rows{4}, num_cols{5};

    float* a = (float*)malloc(sizeof(float) * num_rows * num_cols);
    float* b = (float*)malloc(sizeof(float) * num_rows * num_cols);
    float* c = (float*)malloc(sizeof(float) * num_rows * num_cols);
    initializeMatrix(a, num_rows, num_cols);
    initializeMatrix(b, num_rows, num_cols);

    float *da, *db, *dc;
    cudaMalloc(&da, sizeof(float) * num_rows * num_cols);
    cudaMalloc(&db, sizeof(float) * num_rows * num_cols);
    cudaMalloc(&dc, sizeof(float) * num_rows * num_cols);

    cudaMemcpy(da, a, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock{16, 16, 1};
    dim3 blocksPerGrid{(num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1};
    cudaAdd2DKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, num_rows, num_cols);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost);

    printMatrix(a, num_rows, num_cols);
    printMatrix(b, num_rows, num_cols);
    printMatrix(c, num_rows, num_cols);

    free(a); free(b); free(c);
    cudaFree(a); cudaFree(b); cudaFree(c);

    return;
}

int main(){
    cudaAdd2D();
}