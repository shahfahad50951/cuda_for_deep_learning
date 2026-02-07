#include <iostream>

using namespace std;

void cpuAdd2DKernel(float* a, float* b, float* c, int num_rows, int num_cols){
    for(int row{0}; row < num_rows; row++){
        for(int col{0}; col < num_cols; col++){
            c[row * num_cols + col] = a[row * num_cols + col] + b[row * num_cols + col];
        }
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

void cpuAdd2D(){
    // int num_rows{1000}, num_cols{500};
    int num_rows{4}, num_cols{5};
    float* a = (float*)malloc(sizeof(float) * num_rows * num_cols);
    float* b = (float*)malloc(sizeof(float) * num_rows * num_cols);
    float* c = (float*)malloc(sizeof(float) * num_rows * num_cols);

    initializeMatrix(a, num_rows, num_cols);
    initializeMatrix(b, num_rows, num_cols);

    cpuAdd2DKernel(a, b, c, num_rows, num_cols);

    printMatrix(a, num_rows, num_cols);
    printMatrix(b, num_rows, num_cols);
    printMatrix(c, num_rows, num_cols);
}

int main(){
    cpuAdd2D();
}