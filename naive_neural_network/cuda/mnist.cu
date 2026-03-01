#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <tuple>
#include <cerrno>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

# define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cout << "CUDA call failed. Reason: " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
}

/* GPU KERNEL CODE */
__global__ void transpose_kernel(float* A, float* A_T, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        A_T[col * M + row] = A[row * N + col];
    }
    return;
}

__global__ void gemm_kernel(float* A, float* B, float* C, float* D, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float dotProduct{0.0f};
        for(int k{0}; k < K; k++){
            dotProduct += A[row * K + k] * B[k * N + col];
        }
        if (C != nullptr) dotProduct += C[col];
        D[row * N + col] = dotProduct;
    }
    return;
}

__global__ void relu_kernel(float* A, float* B, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        // Here first f means floating-point max and second f means single-precision floating point
        B[row * N + col] = fmaxf(A[row * N + col], 0.0);
    }
    return;
}

__global__ void softmax_kernel(float* logits, float* probabilities, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float max_val{logits[row * N + 0]};
        for(int i{1}; i < N; i++){
            max_val = fmaxf(max_val, logits[row * N + i]);
        }

        float sum_exp{0.0f};
        for(int i{0}; i < N; i++){
            sum_exp += expf(logits[row * N + i] - max_val);
        }
    
        probabilities[row * N + col] = expf(logits[row * N + col] - max_val) / sum_exp;
    }
    return;
}

__global__ void negativeloglikelihood_kernel(float* probabilities, int* labels, float* nll_value, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row == 0 && col == 0){
        float sum_log_prob{0.0f};
        for(int i{0}; i < M; i++){
            sum_log_prob += logf(probabilities[i * N + labels[i]]);
        }
        sum_log_prob /= M;
        *nll_value = -sum_log_prob;
    }
    return;
}

__global__ void normM_diffone_from_prob_rows_inplace_kernel(float* probabilities, int* labels, int M, int N){
    // For each probablity row, subtract 1.0 from the entry corresponding to correct label and normalize by batch size (M)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        if (labels[row] == col) probabilities[row * N + col] -= 1.0f;
        probabilities[row * N + col] /= M;
    }
    return;
}

__global__ void column_reduce_sum_kernel(float* A, float* col_sum, int M, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < N){
        float sum{0.0f};
        for(int row{0}; row < M; row++){
            sum += A[row * N + col];
        }
        col_sum[col] = sum;
    }
    return;
}

__global__ void relu_derivative_kernel(float* A, float* grad_of_loss_wrt_A, float* grad_of_loss_wrt_B, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        if(A[row * N + col] >= 0.0f) grad_of_loss_wrt_A[row * N + col] = grad_of_loss_wrt_B[row * N + col];
        else grad_of_loss_wrt_A[row * N + col] = 0.0f;
    }
    return;
}

__global__ void sub_inplace_kernel(float* A, float* B, int size, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        A[idx] -= scale * B[idx];
    }
    return;
}


/* CPU KERNEL LAUNCH CODE */
float* transpose(float* A, int M, int N){
    float* A_T;
    CUDA_CHECK(cudaMalloc(&A_T, N * M * sizeof(float)))

    dim3 blockSize{32, 32}; dim3 gridSize{(N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
    transpose_kernel<<<gridSize, blockSize>>>(A, A_T, M, N);
    return A_T;
}

float* gemm(float* A, float* B, float* C, int M, int N, int K){
    float* D;
    CUDA_CHECK(cudaMalloc(&D, sizeof(float) * M * N))

    dim3 blockSize{32, 32};
    dim3 gridSize{(N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
    gemm_kernel<<<gridSize, blockSize>>>(A, B, C, D, M, N, K);
    return D;
}

float* relu(float* A, int M, int N){
    float* B;
    CUDA_CHECK(cudaMalloc(&B, M * N * sizeof(float)))

    dim3 blockSize{32, 32};
    dim3 gridSize{(N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
    relu_kernel<<<gridSize, blockSize>>>(A, B, M, N);
    return B;
}

float* softmax(float* logits, int M, int N){
    float* probabilities;
    CUDA_CHECK(cudaMalloc(&probabilities, M * N * sizeof(float)))
    dim3 blockSize{32, 32};
    dim3 gridSize{(N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
    softmax_kernel<<<gridSize, blockSize>>>(logits, probabilities, M, N);
    return probabilities;
}

float* negativeloglikelihood(float* probabilities, int* labels, int M, int N){
    float* nll_value;
    CUDA_CHECK(cudaMalloc(&nll_value, 1 * sizeof(float)))
    negativeloglikelihood_kernel<<<1,1>>>(probabilities, labels, nll_value, M, N);
    return nll_value;
}

float* cross_entropy_loss(float* logits, int* labels, int M, int N){
    float* probabilities = softmax(logits, M, N);
    float* nll_value = negativeloglikelihood(probabilities, labels, M, N);
    CUDA_CHECK(cudaFree(probabilities))

    return nll_value;
}

float* cross_entropy_loss_derivative(float* logits, int* labels, int M, int N){
    float* probabilities = softmax(logits, M, N);

    dim3 blockSize{32, 32};
    dim3 gridSize{(N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
    normM_diffone_from_prob_rows_inplace_kernel<<<gridSize, blockSize>>>(probabilities, labels, M, N);
    return probabilities;
}

tuple<float*, float*, float*> gemm_derivative(float* A, float* B, float* C, float* grad_of_loss_wrt_D, int M, int N, int K){
    float* B_T = transpose(B, K, N);
    float* grad_of_loss_wrt_A = gemm(grad_of_loss_wrt_D, B_T, nullptr, M, K, N);
    CUDA_CHECK(cudaFree(B_T))

    float* A_T = transpose(A, M, K);
    float* grad_of_loss_wrt_B = gemm(A_T, grad_of_loss_wrt_D, nullptr, K, N, M);
    CUDA_CHECK(cudaFree(A_T))

    float* grad_of_loss_wrt_C = nullptr;
    if(C != nullptr){
        float* col_sum;
        CUDA_CHECK(cudaMalloc(&col_sum, N * sizeof(float)))
        dim3 blockSize{32};
        dim3 gridSize{(N + blockSize.x - 1) / blockSize.x};
        column_reduce_sum_kernel<<<gridSize, blockSize>>>(grad_of_loss_wrt_D, col_sum, M, N);
        grad_of_loss_wrt_C = col_sum;
    }
    return {grad_of_loss_wrt_A, grad_of_loss_wrt_B, grad_of_loss_wrt_C};
}

float* relu_derivative(float* A, float* grad_of_loss_wrt_B, int M, int N){
    float* grad_of_loss_wrt_A;
    CUDA_CHECK(cudaMalloc(&grad_of_loss_wrt_A, M * N * sizeof(float)))

    dim3 blockSize{32, 32}; dim3 gridSize{(N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y};
    relu_derivative_kernel<<<gridSize, blockSize>>>(A, grad_of_loss_wrt_A, grad_of_loss_wrt_B, M, N);
    return grad_of_loss_wrt_A;
}

void sub_inplace(float* A, float* B, int size, float scale){
    dim3 blockSize{32};
    dim3 gridSize{(size + blockSize.x - 1) / blockSize.x};
    sub_inplace_kernel<<<gridSize, blockSize>>>(A, B, size, scale);
    return;
}

/* MODEL DEFINITION CODE */
class MLPConfig {
public:
    int input_size;
    int hidden_size;
    int output_size;
    int train_size;
    int batch_size;
    int epochs;
    float learning_rate;

    MLPConfig(int input_size, int hidden_size, int output_size, int train_size, int batch_size, int epochs, float learning_rate)
    : input_size{input_size},
      hidden_size{hidden_size},
      output_size{output_size},
      train_size{train_size},
      batch_size{batch_size},
      epochs{epochs},
      learning_rate{learning_rate}
    {}
};


class MLP {
public:
    MLPConfig config;
    float *weight1, *weight2, *bias1, *bias2;

    MLP(MLPConfig config)
    : config{config}, weight1{nullptr}, weight2{nullptr}, bias1{nullptr}, bias2{nullptr}
    {
        // Allocate weights and biases in device memory
        CUDA_CHECK(cudaMalloc(&weight1, sizeof(float) * config.input_size * config.hidden_size))
        CUDA_CHECK(cudaMalloc(&weight2, sizeof(float) * config.hidden_size * config.output_size))
        CUDA_CHECK(cudaMalloc(&bias1, sizeof(float) * config.hidden_size))
        CUDA_CHECK(cudaMalloc(&bias2, sizeof(float) * config.output_size))

        // Initialize the allocated weights and biases in device memory
        initialize_weight(weight1, config.input_size, config.hidden_size);
        initialize_weight(weight2, config.hidden_size, config.output_size);
        initialize_bias(bias1, config.hidden_size);
        initialize_bias(bias2, config.output_size);
    }

    ~MLP(){
        CUDA_CHECK(cudaFree(weight1));
        CUDA_CHECK(cudaFree(weight2));
        CUDA_CHECK(cudaFree(bias1));
        CUDA_CHECK(cudaFree(bias2));
    }

    static void initialize_weight(float* weight, int M, int N){
        /* 
            Kaiming-He Initialization
            Initialize data on host for fine-grained control over initialization and then copy over to device
        */
        float* temp = (float*) malloc(M * N * sizeof(float));
        float uniform_dist_range{sqrtf(6.0 / M)};
        for(int row{0}; row < M; row++){
            for(int col{0}; col < N; col++){
                temp[row * N + col] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * uniform_dist_range;
            }
        }

        CUDA_CHECK(cudaMemcpy(weight, temp, M * N * sizeof(float), cudaMemcpyHostToDevice))
        free(temp);
        return;
    }

    static void initialize_bias(float* bias, int N){
        CUDA_CHECK(cudaMemset(bias, 0, N * sizeof(float)));
        return;
    }

    tuple<float*, float*, float*> forward(float* input){
        float* fc1_out = gemm(input, weight1, bias1, config.batch_size, config.hidden_size, config.input_size);
        float* relu_out = relu(fc1_out, config.batch_size, config.hidden_size);
        float* fc2_out = gemm(relu_out, weight2, bias2, config.batch_size, config.output_size, config.hidden_size);

        return {fc1_out, relu_out, fc2_out};
    }

    tuple<float*, float*, float*, float*> backward(float* grad_of_loss_wrt_fc2_out, tuple<float*, float*, float*> cache, float* input){
        float *fc1_out, *relu_out, *fc2_out;
        tie(fc1_out, relu_out, fc2_out) = cache;

        float *grad_of_loss_wrt_relu_out, *grad_of_loss_wrt_fc2_weight, *grad_of_loss_wrt_fc2_bias;
        tie(grad_of_loss_wrt_relu_out, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias) = 
            gemm_derivative(relu_out, weight2, bias2, grad_of_loss_wrt_fc2_out, config.batch_size, config.output_size, config.hidden_size);
        
        float* grad_of_loss_wrt_fc1_out = relu_derivative(fc1_out, grad_of_loss_wrt_relu_out, config.batch_size, config.hidden_size);
        CUDA_CHECK(cudaFree(grad_of_loss_wrt_relu_out))

        float *grad_of_loss_wrt_input, *grad_of_loss_wrt_fc1_weight, *grad_of_loss_wrt_fc1_bias;
        tie(grad_of_loss_wrt_input, grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias) = 
            gemm_derivative(input, weight1, bias1, grad_of_loss_wrt_fc1_out, config.batch_size, config.hidden_size, config.input_size);
        CUDA_CHECK(cudaFree(grad_of_loss_wrt_fc1_out))
        CUDA_CHECK(cudaFree(grad_of_loss_wrt_input))
        
        return {grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias};
    }

    void update_parameters(tuple<float*, float*, float*, float*> gradients){
        float *grad_of_loss_wrt_fc1_weight, *grad_of_loss_wrt_fc1_bias, *grad_of_loss_wrt_fc2_weight, *grad_of_loss_wrt_fc2_bias;
        tie(grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias) = gradients;

        sub_inplace(weight1, grad_of_loss_wrt_fc1_weight, config.input_size * config.hidden_size, config.learning_rate);
        sub_inplace(bias1, grad_of_loss_wrt_fc1_bias, config.hidden_size, config.learning_rate);
        sub_inplace(weight2, grad_of_loss_wrt_fc2_weight, config.hidden_size * config.output_size, config.learning_rate);
        sub_inplace(bias2, grad_of_loss_wrt_fc2_bias, config.output_size, config.learning_rate);
        return;
    }
};


/* TRAINING CODE */
tuple<float*, int*, int, int> load_data_from_binary_file(string directory, MLPConfig config){
    float* x_train = (float*) malloc(config.train_size * config.input_size * sizeof(float));
    int* y_train = (int*) malloc(config.train_size * 1 * sizeof(int));

    fstream x_train_stream{directory + "/x_train.bin", ios::in | ios::binary}; 
    if (!x_train_stream.is_open()){
        cout << "Failed to open file. Exiting...";
        exit(1);
    }
    x_train_stream.read((char*) x_train, config.train_size * config.input_size * sizeof(float));
    x_train_stream.close();

    fstream y_train_stream{directory + "/y_train.bin", ios::in | ios::binary};
    if (!y_train_stream.is_open()){
        cout << "Failed to open file. Exiting...";
        exit(1);
    }
    y_train_stream.read((char*) y_train, config.train_size * sizeof(int));
    y_train_stream.close();

    return {x_train, y_train, config.train_size, config.input_size};
}

tuple<float*, float*> train_timed(MLP& model, MLPConfig config, tuple<float*, int*, int, int>& data){
    // Unpack data related variables
    float* x_train; int *y_train, train_size, input_size;
   tie(x_train, y_train, train_size, input_size)= data;

    float* per_epoch_loss = (float*) malloc(config.epochs * sizeof(float));
    float* timing_stats = (float*) malloc(5 * sizeof(float));
    memset(timing_stats, 0, 5 * sizeof(float));

    // Epoch loop
    for(int epoch{0}; epoch < config.epochs; epoch++){
        // Number of iterations per epoch
        int total_iters{train_size / config.batch_size};
        per_epoch_loss[epoch] = 0.0f;

        // Process all batches one by one
        for(int iter{0}; iter < total_iters; iter++){
            // Create a view over the data to get current batch
            auto before_time = chrono::high_resolution_clock::now();
            int start_sample_idx = iter * config.batch_size;
            float* x_batch = x_train + start_sample_idx * input_size;
            int* y_batch = y_train + start_sample_idx;

            // Move current batch from host to device for processing
            float *device_x_batch; int *device_y_batch;
            CUDA_CHECK(cudaMalloc(&device_x_batch, config.batch_size * input_size * sizeof(float)))
            CUDA_CHECK(cudaMalloc(&device_y_batch, config.batch_size * sizeof(int)))
            CUDA_CHECK(cudaMemcpy(device_x_batch, x_batch, config.batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice))
            CUDA_CHECK(cudaMemcpy(device_y_batch, y_batch, config.batch_size * sizeof(int), cudaMemcpyHostToDevice))
            CUDA_CHECK(cudaDeviceSynchronize())
            auto after_time = chrono::high_resolution_clock::now();
            chrono::duration<double> time_diff{after_time - before_time};
            timing_stats[0] += time_diff.count();

            // Run forward pass of the model on the batch inputs
            // Forward pass caches the intermediate activations because they are required for backward pass
            before_time = chrono::high_resolution_clock::now();
            tuple<float*, float*, float*> cache = model.forward(device_x_batch);
            CUDA_CHECK(cudaDeviceSynchronize())
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[1] += time_diff.count();

            // Compute cross entropy loss from predicted logits and true labels
            before_time = chrono::high_resolution_clock::now();
            float* loss = cross_entropy_loss(get<2>(cache), device_y_batch, config.batch_size, config.output_size);
            CUDA_CHECK(cudaDeviceSynchronize())
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[2] += time_diff.count();

            float host_loss;
            CUDA_CHECK(cudaMemcpy(&host_loss, loss, sizeof(float) * 1, cudaMemcpyDeviceToHost))
            per_epoch_loss[epoch] += host_loss;
            CUDA_CHECK(cudaFree(loss))
            CUDA_CHECK(cudaDeviceSynchronize())

            // Backward pass
            // Compute gradient of loss wrt input to cross entropy loss (fc2_out)
            before_time = chrono::high_resolution_clock::now();
            float* grad_of_loss_wrt_fc2_out = cross_entropy_loss_derivative(get<2>(cache), device_y_batch, config.batch_size, config.output_size);

            // Use chain rule to compute the gradient of loss wrt all the parameters of the model
            auto gradients = model.backward(grad_of_loss_wrt_fc2_out, cache, device_x_batch);
            CUDA_CHECK(cudaDeviceSynchronize())
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[3] += time_diff.count();

            // Free the cached activations (fc1_out, relu_out, fc2_out) and their gradients
            CUDA_CHECK(cudaFree(device_x_batch))
            CUDA_CHECK(cudaFree(device_y_batch))
            CUDA_CHECK(cudaFree(grad_of_loss_wrt_fc2_out))
            CUDA_CHECK(cudaFree(get<0>(cache)))
            CUDA_CHECK(cudaFree(get<1>(cache)))
            CUDA_CHECK(cudaFree(get<2>(cache)))

            // Update the parameters of the model using computed gradients
            before_time = chrono::high_resolution_clock::now();
            model.update_parameters(gradients);
            CUDA_CHECK(cudaDeviceSynchronize())
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[4] += time_diff.count();

            // Free gradients
            CUDA_CHECK(cudaFree(get<0>(gradients)))
            CUDA_CHECK(cudaFree(get<1>(gradients)))
            CUDA_CHECK(cudaFree(get<2>(gradients)))
            CUDA_CHECK(cudaFree(get<3>(gradients)))
        }
        per_epoch_loss[epoch] /= total_iters;
        cout << "Epoch: " << epoch << ", Average Loss: " << per_epoch_loss[epoch] << endl;
    }
    return {per_epoch_loss, timing_stats};
}


/* METRIC REPORTING CODE */
void print_timing_stats(float* timing_stats){
    float total_time{0.0f};
    for(int i{0}; i < 5; i++) total_time += timing_stats[i];

    cout << "Timing Stats\n";
    cout << "\tTotal Training Time: " << total_time << " sec\n\n";
    cout << "Detailed BreakDown\n";
    cout << "\tData Preparation: " << timing_stats[0] << " sec (" << timing_stats[0] * 100 / total_time << "%)\n";
    cout << "\tForward Pass: " << timing_stats[1] << " sec (" << timing_stats[1] * 100 / total_time << "%)\n";
    cout << "\tLoss Computation: " << timing_stats[2] << " sec (" << timing_stats[2] * 100 / total_time << "%)\n";
    cout << "\tBackward Pass: " << timing_stats[3] << " sec (" << timing_stats[3] * 100 / total_time << "%)\n";
    cout << "\tParameters Update: " << timing_stats[4] << " sec (" << timing_stats[4] * 100 / total_time << "%)\n";
    cout << endl;
}

int main(){
    // Set custom seed for reproducability across C++ and CUDA impelementation
    srand(1U);

    constexpr int INPUT_SIZE{784};
    constexpr int HIDDEN_SIZE{256};
    constexpr int OUTPUT_SIZE{10};
    constexpr int TRAIN_SIZE{10000};
    constexpr int BATCH_SIZE{8};
    constexpr int EPOCHS{10};
    constexpr float LEARNING_RATE{0.01f};

    MLPConfig config{INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TRAIN_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE};
    MLP model(config);

    tuple<float*, int*, int, int> data = load_data_from_binary_file("../utils/data", config);

    // Start training loop
    float *per_epoch_loss, *timing_stats;
    tie(per_epoch_loss, timing_stats) = train_timed(model, config, data);
    print_timing_stats(timing_stats);

    // Free training data
    free(get<0>(data)); free(get<1>(data));
    return 0;

}