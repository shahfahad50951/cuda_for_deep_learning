/*
    Implements standard CPU and GPU GEMM using CUTE Tensor, Shape, Layout, and Stride
    abstractions. The GPU GEMM is a non-tiled implementation with one thread owning
    one output element; for an (M,N) output matrix, M * N threads are launched.
*/

#include <iostream>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>

using namespace std;
using cute::make_tensor, cute::make_layout;
using cute::make_shape, cute::make_stride;
using cute::print_layout, cute::print_tensor;
using cute::layout, cute::shape, cute::stride;

template <class Tensor>
void initializeMatrixConst(Tensor& matrix, float initVal = 0.0f){
    for(int m{0}; m < cute::size<0>(matrix); m++){
        for(int n{0}; n < cute::size<1>(matrix); n++){
            matrix(m,n) = initVal;
        }
    }
    return;
}

template <class Tensor>
void initializeMatrixSeq(Tensor& matrix){
    for(int m{0}; m < cute::size<0>(matrix); m++){
        for(int n{0}; n < cute::size<1>(matrix); n++){
            matrix(m,n) = min(64.0f, (float) m * get<0>(stride(matrix)) + (float) n * get<1>(stride(matrix)));
        }
    }
    return;
}

template <class TensorA, class TensorB>
bool areEqual(TensorA A, TensorB B, float atol=1e-3){
    bool equal{true};
    for(int m{0}; m < size<0>(A); m++){
        for(int n{0}; n < size<1>(A); n++){
            if(abs(A(m, n) - B(m, n)) > atol){
                equal = false;
                break;
            }
        }
    }
    return equal;
}

template <class TensorA, class TensorB, class TensorC>
void cpu_gemm(const TensorA& A, const TensorB& B, TensorC C){
    for(int m{0}; m < size<0>(C); m++){
        for(int n{0}; n < size<1>(C); n++){
            float dotprod = 0.0f;
            for(int k{0}; k < size<1>(A); k++){
                dotprod += A(m, k) * B(k, n);
            }
            C(m, n) = dotprod;
        }
    }
    return;
}

template <class TensorA, class TensorB, class TensorC>
__global__ void device_gemm(TensorA A, TensorB B, TensorC C){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if(m < size<0>(C) && n < size<1>(C)){
        float dotprod = 0.0;
        for(int k{0}; k < size<1>(A); k++){
            dotprod += A(m, k) * B(k, n);
        }
        C(m,n) = C(m,n) + dotprod;
    }
    return;
}


int main(){
    // Problem dimensions
    int M{1024}, N{1024}, K{1024};

    /* CPU GEMM START */

    // Raw 1D storage
    float* A_data = new float[M*K];
    float* B_data = new float[K*N];
    float* C_data = new float[M*N];

    // Create Layouts
    auto ALayout = make_layout(make_shape(M, K), make_stride(K, 1)); // K-major
    auto BLayout = make_layout(make_shape(K, N), make_stride(1, K)); // K-major
    auto CLayout = make_layout(make_shape(M, N), make_stride(N, 1)); // N-major

    // Create Tensors
    auto ATensor = make_tensor(A_data, ALayout);
    auto BTensor = make_tensor(B_data, BLayout);
    auto CTensor = make_tensor(C_data, CLayout);

    // Initialize Matrices
    initializeMatrixSeq(ATensor);
    initializeMatrixSeq(BTensor);
    initializeMatrixConst(CTensor, 0.0f);

    // Matrix Multiplication
    cpu_gemm(ATensor, BTensor, CTensor);

    /* CPU GEMM END */

    /* GPU GEMM START */
    
    // Allocate raw gpu storage
    float *A_gpudata, *B_gpudata, *C_gpudata; 
    cudaMalloc(&A_gpudata, M * K * sizeof(float));
    cudaMalloc(&B_gpudata, K * N * sizeof(float));
    cudaMalloc(&C_gpudata, M * N * sizeof(float));

    // Create Layouts
    auto ALayoutGPU = make_layout(make_shape(M, K), make_stride(1, M));  // M-major
    auto BLayoutGPU = make_layout(make_shape(K, N), make_stride(1, K));  // K-major
    auto CLayoutGPU = make_layout(make_shape(M, N), make_stride(1, M));  // M-major

    // Create Tensors
    auto ATensorGPU = make_tensor(A_gpudata, ALayoutGPU);
    auto BTensorGPU = make_tensor(B_gpudata, BLayoutGPU);
    auto CTensorGPU = make_tensor(C_gpudata, CLayoutGPU);

    // Initialize device tensors 
    // When copying raw data, make sure that layouts of source and destination are same
    // otherwise the interpretation of source and destination raw data will change
    float* A_data_gpu_compatible = new float[M * K];
    auto ATensorGPUCompatible = make_tensor(A_data_gpu_compatible, ALayoutGPU);
    for(int m{0}; m < size<0>(ATensor); m++){
        for(int n{0}; n < size<1>(ATensor); n++) ATensorGPUCompatible(m,n) = ATensor(m,n);
    }

    cudaMemcpy(A_gpudata, A_data_gpu_compatible, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpudata, B_data, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C_gpudata, 0, M * N * sizeof(float));

    delete[] A_data_gpu_compatible;

    // Kernel Launch
    dim3 blockSize{32, 32, 1};
    dim3 gridSize{(M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, 1};
    device_gemm<<<gridSize, blockSize>>>(ATensorGPU, BTensorGPU, CTensorGPU);
    cudaDeviceSynchronize();

    // Inspect the result on host
    float* C_gpudata_copy = new float[M * N];
    cudaMemcpy(C_gpudata_copy, C_gpudata, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    auto CTensor_copy = make_tensor(C_gpudata_copy, CLayoutGPU);
    bool equal = areEqual(CTensor, CTensor_copy);

    if(equal) cout << "Result of GEMM on CPU and GPU match!\n";
    else  cout << "Result of GEMM on CPU and GPU DO NOT match\n";

    /* GPU GEMM END */


    // Free raw cpu 1D storage
    delete[] A_data; delete[] B_data; delete[] C_data; delete[] C_gpudata_copy;

    // Free raw device 1D storage
    cudaFree(A_gpudata); cudaFree(B_gpudata); cudaFree(C_gpudata);

    return 0;
}

/*
    Compilation Command:
    nvcc -I ./cutlass/include --std=c++17 basic_gemm.cu
*/