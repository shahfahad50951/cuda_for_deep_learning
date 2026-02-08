#include <iostream>
#include <cmath>

using namespace std;

void initializeArray(float* a, int size){
    for(int i{0}; i < size; i++) a[i] = i;
    return;
}

void printArray(float* a, int size){
    for(int i{0}; i < size; i++) cout << a[i] << ' ';
    cout << '\n';
    return;
}

void conv1dKernel(float* input, float* output, float* kernel, int inputSize, int kernelSize){
    int outputSize = inputSize - kernelSize + 1;
    for(int i{0}; i < outputSize; i++){
        float dotProduct{0.0f};
        for(int k{0}; k < kernelSize; k++){
            dotProduct += input[i + k] * kernel[k];
        }
        output[i] = dotProduct;
    }
    return;
}

void conv1d(){
    int inputSize{20}, kernelSize{3};
    int outputSize{inputSize - kernelSize + 1};

    float* in = (float*) malloc(sizeof(float) * inputSize);
    float* out = (float*) malloc(sizeof(float) * outputSize);
    float* kernel = (float*) malloc(sizeof(float) * kernelSize);

    initializeArray(in, inputSize);
    initializeArray(kernel, kernelSize);

    conv1dKernel(in, out, kernel, inputSize, kernelSize);

    printArray(in, inputSize);
    printArray(kernel, kernelSize);
    printArray(out, outputSize);

    free(in); free(out); free(kernel);

    return;
}

int main(){
    conv1d();
}