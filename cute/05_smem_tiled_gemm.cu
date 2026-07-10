#include <iostream>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>

using namespace std;
using cute::make_layout, cute::make_tensor, cute::make_shape, cute::make_stride;
using cute::stride, cute::size;
using cute::print_layout, cute::print_tensor;
using cute::zipped_divide;
using cute::make_coord;

template<class Tensor>
void initializeMatrixConst(Tensor M, float initVal = 0.0f){
    for(int r{0}; r < size<0>(M); r++){
        for(int c{0}; c < size<1>(M); c++){
            M(r,c) = initVal;
        }
    }
    return;
}

template<class Tensor>
void initializeMatrixSeq(Tensor M){
    for(int r{0}; r < size<0>(M); r++){
        for(int c{0}; c < size<1>(M); c++){
            M(r,c) = min(64.0f, (float) r * get<0>(stride(M)) + (float) c * get<1>(stride(M)));
        }
    }
    return;
}

template<class TensorA, class TensorB, class TensorC>
void cpu_gemm(TensorA A, TensorB B, TensorC C){
    for(int r{0}; r < size<0>(C); r++){
        for(int c{0}; c < size<1>(C); c++){
            for(int k{0}; k < size<1>(A); k++){
                C(r,c) += A(r,k) * B(k, c);
            }
        }
    }
    return;
}

template<class TensorA, class TensorB>
bool areEqual(TensorA A, TensorB B, float atol=1e-3){
    int totalIncorrect = 0;
    float largestMismatch = 0.0f;
    for(int r{0}; r < size<0>(A); r++){
        for(int c{0}; c < size<1>(B); c++){
            if(abs(A(r,c) - B(r,c)) > atol) {
                ++totalIncorrect;
                largestMismatch = max(largestMismatch, abs(A(r,c) - B(r,c)));
            }
        }
    }

    if(totalIncorrect > 0){
        cout << "Total failed matches: " << totalIncorrect << '\n';
        cout << "Largest difference between values: " << largestMismatch << '\n';
    }

    return (totalIncorrect == 0) ? true : false;
}


template<class TensorA, class TensorB, class TensorC, class BLK_M_t, class BLK_N_t, class BLK_K_t>
__global__ void gpu_gemm(TensorA A, TensorB B, TensorC C, BLK_M_t  BLK_M, BLK_N_t BLK_N, BLK_K_t BLK_K){
    // Find current CTA's index within the launch grid
    int gridx = blockIdx.x, gridy = blockIdx.y;

    // Find current thread's location within local CTA
    int lx = threadIdx.x, ly = threadIdx.y;

    // Find current thread's location within complete launch grid
    int gx = gridx * blockDim.x + lx;
    int gy = gridy * blockDim.y + ly;

    // Problem dimensions
    int M = size<0>(C), N = size<1>(C), K = size<1>(A);

    // Allocate SMEM for the A and B chunks
    __shared__ float ASmem[BLK_M * BLK_K];
    __shared__ float BSmem[BLK_K * BLK_N];

    // Create SMEM layout for A and B chunks
    auto ALayoutSmem = make_layout(make_shape(BLK_M, BLK_K), make_stride(1, BLK_M)); // M-major SMEM layout
    auto BLayoutSmem = make_layout(make_shape(BLK_K, BLK_N), make_stride(1, BLK_K)); // K-major SMEM layout

    // Create SMEM tensors for A and B chunks
    auto ATensorSmem = make_tensor(&ASmem[0], ALayoutSmem);
    auto BTensorSmem = make_tensor(&BSmem[0], BLayoutSmem);

    // Make new view of the global memory tensors A and B which are tiled views with
    // modes (within_tile, which_tile)
    auto gmemTiledATensor = zipped_divide(A, make_shape(BLK_M, BLK_K));
    auto gmemTiledBTensor = zipped_divide(B, make_shape(BLK_K, BLK_N));

    // Slice out all K-tiles of A and B which current CTA's threads will use
    auto gmemATensorAllKChunks = gmemTiledATensor(cute::_, make_coord(gridx, cute::_));
    auto gmemBTensorAllKChunks = gmemTiledBTensor(cute::_, make_coord(cute::_, gridy));

    // Total number of BLK_K chunks to process
    int num_k_chunks = (K + BLK_K - 1) / BLK_K;

    // Register accumulator for current thread's output across all k_chunks
    float dotProduct = 0.0f;

    for(int k_chunk{0}; k_chunk < num_k_chunks; k_chunk++){
        // Load the current A[BLK_M x BLK_K] chunk from GMEM into SMEM
        auto gmemATensor = gmemATensorAllKChunks(make_coord(cute::_,cute::_), k_chunk);

        // At a time, CTA (BLK_M, BLK_N) will load (BLK_M x BLK_N) elements. However, we need to load
        // (BLK_M, BLK_K) A chunk. Loading entire BLK_K might require multiple iterations along the BLK_K
        // dimension with each step loading BLK_N chunk of BLK_K dimension, thus requiring
        // ceil(BLK_K / BLK_N) iterations
        int num_iters_to_load_full_A_chunk = (BLK_K + BLK_N - 1) / BLK_N;
        for(int i{0}; i < num_iters_to_load_full_A_chunk; i++){
            // If BLK_K is not perfectly divisible by BLK_N, then in the last iteration the size along the
            // BLK_K dimension that needs to be loaded would be BLK_K % BLK_N
            if((BLK_K % BLK_N != 0) && (i == num_iters_to_load_full_A_chunk - 1)){
                int remaining_k = BLK_K % BLK_N;
                if(ly < remaining_k) ATensorSmem(lx, i * BLK_N + ly) = gmemATensor(lx, i * BLK_N + ly);
            }
            else
                ATensorSmem(lx, i * BLK_N + ly) = gmemATensor(lx, i * BLK_N + ly);
        }

        // Load the current B[BLK_K x BLK_N] chunk from GMEM into SMEM
        auto gmemBTensor = gmemBTensorAllKChunks(make_coord(cute::_,cute::_), k_chunk);

        // At a time, CTA (BLK_M, BLK_N) will load (BLK_M x BLK_N) elements. However, we need to load
        // (BLK_K, BLK_N) B chunk. Loading entire BLK_K might require multiple iterations along the BLK_K
        // dimension with each step loading BLK_M chunk of BLK_K dimension, thus requiring ceil(BLK_K / BLK_M)
        // iterations
        int num_iters_to_load_full_B_chunk = (BLK_K + BLK_M - 1) / BLK_M;
        for(int i{0}; i < num_iters_to_load_full_B_chunk; i++){
            // If BLK_K is not perfectly divisible by BLK_M, then in the last iteration the size along the
            // BLK_K dimension that needs to be loaded would be BLK_K % BLK_M
            if((BLK_K % BLK_M != 0) && (i == num_iters_to_load_full_B_chunk - 1)){
                int remaining_k = BLK_K % BLK_M;
                if(lx < remaining_k) BTensorSmem(i * BLK_M + lx, ly) = gmemBTensor(i * BLK_M + lx, ly);
            }
            else
                BTensorSmem(i * BLK_M + lx, ly) = gmemBTensor(i * BLK_M + lx, ly);
        }

        // Wait for all threads to load their corresponding data to SMEM tensors and then proceed
        __syncthreads();

        // Compute dot product for the current thread's assigned output element (of C), using the loaded A
        // and B SMEM chunks
        for(int k{0}; k < BLK_K; k++){
            dotProduct += ATensorSmem(lx, k) * BTensorSmem(k, ly);
        }

        // Wait for all threads to complete their k iterations before moving on to load SMEM tensors
        // for next k_chunk
        __syncthreads();
    }

    // Read the global C element assigned to this thread for computation, add it to the accumulated dot product
    // then store it back to GMEM
    C(gx, gy) = C(gx, gy) + dotProduct;

    return;
}

/*
Assumption: M, N and K are divisible by BLK_M, BLK_N, BLK_K respectively
*/

int main(){
    // int M = 1024, N = 1024, K = 1024;
    // int M = 64, N = 64, K = 96;
    int M = 64, N = 64, K = 192;

    /* CPU GEMM START */

    // Create cpu layouts of tensors A, B and C
    auto ALayout = make_layout(make_shape(M, K), make_stride(K, 1));
    auto BLayout = make_layout(make_shape(K, N), make_stride(1, K));
    auto CLayout = make_layout(make_shape(M, N), make_stride(N, 1));

    // Allocate raw 1D-memory for cute tensors
    float* AData = new float[size(ALayout)];
    float* BData = new float[size(BLayout)];
    float* CData = new float[size(CLayout)];

    // Create cpu tensors for A, B and C
    auto ATensor = make_tensor(AData, ALayout);
    auto BTensor = make_tensor(BData, BLayout);
    auto CTensor = make_tensor(CData, CLayout);

    // Initialize cpu tensors
    // initializeMatrixSeq(ATensor); initializeMatrixSeq(BTensor);
    initializeMatrixConst(ATensor, 1.0f); initializeMatrixSeq(BTensor);
    initializeMatrixConst(CTensor, 0.0f);

    // Call cpu_gemm to perform gemm operation
    cpu_gemm(ATensor, BTensor, CTensor);

    // Print tensors for manual verification
    // print_tensor(ATensor); print_tensor(BTensor); print_tensor(CTensor);

    /* CPU GEMM END */


    /* GPU GEMM START */

    // Create gpu layouts of tensors A, B and C
    // These layouts are optimized to enable global memory coalescing during global memory access
    auto ALayoutGPU = make_layout(make_shape(M, K), make_stride(1, M)); // M-major
    auto BLayoutGPU = make_layout(make_shape(K, N), make_stride(1, K)); // K-major
    auto CLayoutGPU = make_layout(make_shape(M, N), make_stride(1, M)); // M-major

    // Allocate raw 1D-memory for cute gpu tensors
    float *ADataGPU, *BDataGPU, *CDataGPU;
    cudaMalloc(&ADataGPU, size(ALayoutGPU) * sizeof(float));
    cudaMalloc(&BDataGPU, size(BLayoutGPU) * sizeof(float));
    cudaMalloc(&CDataGPU, size(CLayoutGPU) * sizeof(float));

    // Create gpu tensors A, B and C
    auto ATensorGPU = make_tensor(ADataGPU, ALayoutGPU);
    auto BTensorGPU = make_tensor(BDataGPU, BLayoutGPU);
    auto CTensorGPU = make_tensor(CDataGPU, CLayoutGPU);

    // Initialize gpu tensors
    // Create a cpu tensor with the same layout as GPU tensor so that we can copy the underlying data
    // to the GPU. Without this, raw data copy of the ATensor on cpu to ATensor on GPU will be incorrect
    float* ADataGPUCompatible = new float[size(ATensorGPU)];
    auto ATensorGPUCompatible = make_tensor(ADataGPUCompatible, ALayoutGPU);
    for(int r{0}; r < size<0>(ATensorGPUCompatible); r++){
        for(int c{0}; c < size<1>(ATensorGPUCompatible); c++) ATensorGPUCompatible(r,c) = ATensor(r,c);
    }

    cudaMemcpy(ADataGPU, ADataGPUCompatible, size(ATensorGPUCompatible) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(BDataGPU, BData, size(BTensor) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(CDataGPU, 0, size(CTensorGPU) * sizeof(float));

    // Setup gpu_gemm kernel launch parameters
    auto BLK_M = cute::_32{}; auto BLK_N = cute::_32{}; auto BLK_K = cute::_96{};
    dim3 blockSize{BLK_M, BLK_N};
    dim3 gridSize{(M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y};

    // Call gpu_gemm to perform gemm operation on GPU
    gpu_gemm<<<gridSize, blockSize>>>(ATensorGPU, BTensorGPU, CTensorGPU, BLK_M, BLK_N, BLK_K);
    cudaDeviceSynchronize();

    /* GPU GEMM END */


    // Compare gpu gemm result with cpu gemm for correctness

    // Create a cpu tensor, move results from GPU to CPU
    float* CDataGPUResult = new float[size(CTensorGPU)];
    auto CTensorGPUResult = make_tensor(CDataGPUResult, CLayoutGPU);
    cudaMemcpy(CDataGPUResult, CDataGPU, size(CTensorGPU) * sizeof(float), cudaMemcpyDeviceToHost);
    // print_tensor(CTensorGPUResult);

    // Compare cpu and gpu gemm results
    if(areEqual(CTensor, CTensorGPUResult))
        cout << "Result of CPU and GPU GEMM match!\n";
    else
        cout << "Results of CPU and GPU GEMM DO NOT match\n";

    // Release cpu and gpu memory
    delete[] AData; delete[] BData; delete[] CData; delete[] ADataGPUCompatible; delete[] CDataGPUResult;
    cudaFree(ADataGPU); cudaFree(BDataGPU); cudaFree(CDataGPU);

    return 0;
}

/*
    Compilation Command:
    nvcc -lineinfo -I ./cutlass/include --std=c++17 -o smem_tiled_gemm 05_smem_tiled_gemm.cu
*/
