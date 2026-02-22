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

using namespace std;

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

float* transpose(float* A, int M, int N){
    float* A_T = (float*) malloc(N * M * sizeof(float));
    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            A_T[col * M + row] = A[row * N + col];
        }
    }
    return A_T;
}

float* gemm(float* A, float* B, float* C, int M, int N, int K){
    /* 
    Implements standard GEMM operation: D = A @ B + C
    A = M x K, B = K x N, C = N, D = M x N
    */
    float* D = (float*) malloc(M * N * sizeof(float));
    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            float dotProduct{0.0f};
            for(int k{0}; k < K; k++) dotProduct += A[row * K + k] * B[k * N + col];
            if (C != nullptr) dotProduct += C[col];
            D[row * N + col] = dotProduct;
        }
    }
    return D;
}

tuple<float*, float*, float*> gemm_derivative(float* A, float* B, float* C, float* grad_of_loss_wrt_D, int M, int N, int K){
    float *A_T = transpose(A, M, K), *B_T = transpose(B, K, N);
    float* grad_of_loss_wrt_A = gemm(grad_of_loss_wrt_D, B_T, nullptr, M, K, N);
    float* grad_of_loss_wrt_B = gemm(A_T, grad_of_loss_wrt_D, nullptr, K, N, M);

    free(A_T); free(B_T);

    float* grad_of_loss_wrt_C = nullptr;
    if(C != nullptr){
        grad_of_loss_wrt_C = (float*) malloc(N * sizeof(float));
        for(int row{0}; row < M; row++){
            for(int col{0}; col < N; col++){
                if(row == 0) grad_of_loss_wrt_C[col] = 0;
                grad_of_loss_wrt_C[col] += grad_of_loss_wrt_D[row * N + col];
            }
        }
    }

    return {grad_of_loss_wrt_A, grad_of_loss_wrt_B, grad_of_loss_wrt_C};
}

float* relu(float* A, int M, int N){
    float* B = (float*) malloc(M * N * sizeof(float));
    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            B[row * N + col] = max(A[row * N + col], 0.0f);
        }
    }
    return B;
}

float* relu_derivative(float* A, float* grad_of_loss_wrt_B, int M, int N){
    float* grad_of_loss_wrt_A = (float*) malloc(M * N * sizeof(float));
    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            if(A[row * N + col] >= 0) grad_of_loss_wrt_A[row * N + col] = grad_of_loss_wrt_B[row * N + col];
            else grad_of_loss_wrt_A[row * N + col] = 0;
        }
    }
    return grad_of_loss_wrt_A;
}

float* softmax(float* A, int M, int N){
    float* row_max = (float*) malloc(M * sizeof(float));
    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            if(col == 0) row_max[row] = A[row * N + col];
            row_max[row] = max(row_max[row], A[row * N + col]);
        }
    }

    float* row_exp_sum = (float*) malloc(M * sizeof(float));
    for(int row{0}; row < M; row++){
        row_exp_sum[row] = 0;
        for(int col{0}; col < N; col++){
            row_exp_sum[row] += expf(A[row * N + col] - row_max[row]);
        }
    }

    float* probabilities = (float*) malloc(M * N * sizeof(float));
    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            probabilities[row * N + col] = expf(A[row * N + col] - row_max[row]) / row_exp_sum[row];
        }
    }

    free(row_max); free(row_exp_sum);

    return probabilities;
}

float cross_entropy_loss(float* y_pred, int* labels, int M, int N){
    float* probabilities = softmax(y_pred, M, N);

    float sum_log_prob{0.0f};
    for(int row{0}; row < M; row++){
        sum_log_prob += logf(probabilities[row * N + labels[row]]);
    }
    float loss = (-sum_log_prob) / M;
    free(probabilities);
    return loss;
}

float* cross_entropy_loss_derivative(float* fc2_out, int* labels, int M, int N){
    float* grad_of_loss_wrt_fc2_out = (float*) malloc(M * N * sizeof(float));
    float* probabilities = softmax(fc2_out, M, N);

    for(int row{0}; row < M; row++){
        for(int col{0}; col < N; col++){
            float to_sub{col == labels[row] ? 1.0f : 0.0f};
            grad_of_loss_wrt_fc2_out[row * N + col] = (probabilities[row * N + col] - to_sub) / M;
        }
    }
    free(probabilities);
    return grad_of_loss_wrt_fc2_out;
}

void elementwise_diff_inplace(float* A, float* B, int size, float scale){
    for(int i{0}; i < size; i++){
        A[i] -= scale * B[i];
    }
    return;
}

class MLP {
public:
    MLPConfig config;
    float *weight1, *weight2, *bias1, *bias2;

    MLP(MLPConfig config)
    : config{config}, weight1{nullptr}, weight2{nullptr}, bias1{nullptr}, bias2{nullptr}
    {
        weight1 = (float*) malloc(config.input_size * config.hidden_size * sizeof(float));
        weight2 = (float*) malloc(config.hidden_size * config.output_size * sizeof(float));
        bias1 = (float*) malloc(config.hidden_size * sizeof(float));
        bias2 = (float*) malloc(config.output_size * sizeof(float));

        initialize_weight(weight1, config.input_size, config.hidden_size);
        initialize_weight(weight2, config.hidden_size, config.output_size);
        initialize_bias(bias1, config.hidden_size);
        initialize_bias(bias2, config.output_size);
    }

    ~MLP(){
        free(weight1);
        free(weight2);
        free(bias1);
        free(bias2);
    }

    static void initialize_weight(float* weight, int M, int N){
        // Kaiming-He Initialization
        float uniform_dist_range{sqrtf(6.0 / M)};
        for(int i{0}; i < M; i++){
            for(int j{0}; j < N; j++){
                weight[i * N + j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * uniform_dist_range;
            }
        }
        return;
    }

    static void initialize_bias(float* bias, int N){
        memset(bias, 0, sizeof(float) * N);
    }

    tuple<float*, float*, float*> forward(float* input){
        float* fc1_out = gemm(input, weight1, bias1, config.batch_size, config.hidden_size, config.input_size);
        float* relu_out = relu(fc1_out, config.batch_size, config.hidden_size);
        float* fc2_out = gemm(relu_out, weight2, bias2, config.batch_size, config.output_size, config.hidden_size);
        return {fc1_out, relu_out, fc2_out};
    }

    tuple<float*, float*, float*, float*> backward(float* grad_of_loss_wrt_fc2_out, tuple<float*, float*, float*> cache, float* input){
        auto [fc1_out, relu_out, fc2_out] = cache;

        auto [grad_of_loss_wrt_relu_out, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias] = 
            gemm_derivative(relu_out, weight2, bias2, grad_of_loss_wrt_fc2_out, config.batch_size, config.output_size, config.hidden_size);

        float* grad_of_loss_wrt_fc1_out = relu_derivative(fc1_out, grad_of_loss_wrt_relu_out, config.batch_size, config.hidden_size);

        auto [grad_of_loss_wrt_input, grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias] = 
            gemm_derivative(input, weight1, bias1, grad_of_loss_wrt_fc1_out, config.batch_size, config.hidden_size, config.input_size);
        
        return {grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias};
    }

    void update_parameters(tuple<float*, float*, float*, float*> gradients){
        auto [grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias] =
            gradients;

        elementwise_diff_inplace(weight1, grad_of_loss_wrt_fc1_weight, config.input_size * config.hidden_size, config.learning_rate);
        elementwise_diff_inplace(bias1, grad_of_loss_wrt_fc1_bias, config.hidden_size, config.learning_rate);
        elementwise_diff_inplace(weight2, grad_of_loss_wrt_fc2_weight, config.hidden_size * config.output_size, config.learning_rate);
        elementwise_diff_inplace(bias2, grad_of_loss_wrt_fc2_bias, config.output_size, config.learning_rate);
        return;
    }
};

tuple<float*, float*> train_timed(MLP& model, MLPConfig config, tuple<float*, int*, int, int>& data){
    // Unpack data related variables
    auto [x_train, y_train, train_size, input_size] = data;

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
            auto after_time = chrono::high_resolution_clock::now();
            chrono::duration<double> time_diff{after_time - before_time};
            timing_stats[0] += time_diff.count();

            // Run forward pass of the model on the batch inputs
            // Forward pass caches the intermediate activations because they are required for backward pass
            before_time = chrono::high_resolution_clock::now();
            tuple<float*, float*, float*> cache = model.forward(x_batch);
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[1] += time_diff.count();


            // Compute cross entropy loss from predicted logits and true labels
            before_time = chrono::high_resolution_clock::now();
            float loss = cross_entropy_loss(get<2>(cache), y_batch, config.batch_size, config.output_size);
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[2] += time_diff.count();
            per_epoch_loss[epoch] += loss;

            // Backward pass
            // Compute gradient of loss wrt input to cross entropy loss (fc2_out)
            before_time = chrono::high_resolution_clock::now();
            float* grad_of_loss_wrt_fc2_out = cross_entropy_loss_derivative(get<2>(cache), y_batch, config.batch_size, config.output_size);
            // Use chain rule to compute the gradient of loss wrt all the parameters of the model
            auto gradients = model.backward(grad_of_loss_wrt_fc2_out, cache, x_batch);
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[3] += time_diff.count();

            // Free the cached activations
            free(get<0>(cache)); free(get<1>(cache)); free(get<2>(cache));

            // Update the parameters of the model using computed gradients
            before_time = chrono::high_resolution_clock::now();
            model.update_parameters(gradients);
            after_time = chrono::high_resolution_clock::now();
            time_diff = after_time - before_time;
            timing_stats[4] += time_diff.count();

            // Free gradients
            free(get<0>(gradients)); free(get<1>(gradients)); free(get<2>(gradients)); free(get<3>(gradients));
        }
        per_epoch_loss[epoch] /= total_iters;
        cout << "Epoch: " << epoch << ", Average Loss: " << per_epoch_loss[epoch] << endl;
    }
    return {per_epoch_loss, timing_stats};
}

void print_timing_stats(float* timing_stats){
    float total_time{0.0f};
    for(int i{0}; i < 5; i++) total_time += timing_stats[i];

    cout << "Timing Stats\n";
    cout << "\tData Preparation: " << timing_stats[0] << " sec (" << timing_stats[0] * 100 / total_time << "%)\n";
    cout << "\tForward Pass: " << timing_stats[1] << " sec (" << timing_stats[1] * 100 / total_time << "%)\n";
    cout << "\tLoss Computation: " << timing_stats[2] << " sec (" << timing_stats[2] * 100 / total_time << "%)\n";
    cout << "\tBackward Pass: " << timing_stats[3] << " sec (" << timing_stats[3] * 100 / total_time << "%)\n";
    cout << "\tParameters Update: " << timing_stats[4] << " sec (" << timing_stats[4] * 100 / total_time << "%)\n";
    cout << endl;
}

int main(){
    constexpr int INPUT_SIZE{784};
    constexpr int HIDDEN_SIZE{256};
    constexpr int OUTPUT_SIZE{10};
    constexpr int TRAIN_SIZE{10000};
    constexpr int BATCH_SIZE{8};
    constexpr int EPOCHS{10};
    constexpr float LEARNING_RATE{0.01f};

    MLPConfig config{INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TRAIN_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE};
    MLP model(config);

    tuple<float*, int*, int, int> data = 
        load_data_from_binary_file("../utils/data", config);
    
    // Start training loop
    auto [per_epoch_loss, timing_stats] = train_timed(model, config, data);
    print_timing_stats(timing_stats);

    // Free training data
    free(get<0>(data)); free(get<1>(data));
    return 0;

}