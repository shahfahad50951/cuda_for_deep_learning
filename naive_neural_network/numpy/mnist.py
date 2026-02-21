import time
import numpy as np

# DataLoading
def load_data_from_binary_files(dataset_dir, TRAIN_SIZE=10000, TEST_SIZE=200):
    x_train_path = f'{dataset_dir}/x_train.bin'
    y_train_path = f'{dataset_dir}/y_train.bin'
    x_test_path = f'{dataset_dir}/x_test.bin'
    y_test_path = f'{dataset_dir}/y_test.bin'

    x_train = np.fromfile(x_train_path, dtype=np.float32).reshape(60000, 784)[:TRAIN_SIZE]
    y_train = np.fromfile(y_train_path, dtype=np.int32).reshape(60000)[:TRAIN_SIZE]
    x_test = np.fromfile(x_test_path, dtype=np.float32).reshape(10000, 784)[:TEST_SIZE]
    y_test = np.fromfile(y_test_path, dtype=np.int32).reshape(10000)[:TEST_SIZE]
    return x_train, y_train, x_test, y_test

# Print training stats
def print_stats(timing_stats):
    total_time = 0.0
    for key in timing_stats: total_time += timing_stats[key]
    print(f'Total Training Time: {total_time:.2f}')
    print('Detailed Breakdown')
    for key in timing_stats: 
        stage, time_taken = key, timing_stats[key]
        pct_time_taken = time_taken * 100 / total_time
        print(f'\t{stage}: {time_taken:.2f} ({pct_time_taken:.2f}%)')
    return

def relu(input: np.ndarray):
    # Elementwise maximum
    output = np.maximum(input, 0)
    return output

def relu_derivative(input, grad_of_loss_wrt_output):
    # Conditional selection
    return np.where(input > 0, grad_of_loss_wrt_output, 0)

def linear(input, weight, bias):
    # Linear layer operation implementation
    return input @ weight + bias

def linear_derivative(input, weight, grad_of_loss_wrt_output):
    grad_of_loss_wrt_weight = input.T @ grad_of_loss_wrt_output
    grad_of_loss_wrt_input = grad_of_loss_wrt_output @ weight.T
    grad_of_loss_wrt_bias = np.sum(grad_of_loss_wrt_output, axis=0)
    return grad_of_loss_wrt_input, grad_of_loss_wrt_weight, grad_of_loss_wrt_bias

def softmax(input):
    max_val = np.max(input, axis=1, keepdims=True)
    exp_input = np.exp(input-max_val)
    sum_exp_input = np.sum(exp_input, axis=1, keepdims=True)
    return exp_input / sum_exp_input

def cross_entropy_loss(pred_logits, labels):
    probabilities = softmax(pred_logits)
    batch_size = pred_logits.shape[0]
    # Advanced Indexing: Fancy Indexing: Indexing with integer lists
    correct_class_log_probabilities = np.log(probabilities[np.arange(batch_size), labels])
    negative_log_likelihood = -np.sum(correct_class_log_probabilities) / batch_size
    return negative_log_likelihood

def cross_entropy_loss_derivative(pred_logits, labels):
    batch_size, num_classes = pred_logits.shape
    probabilities = softmax(pred_logits)
    one_hot_encoded_labels = np.zeros((batch_size, num_classes)).astype(np.float32)
    one_hot_encoded_labels[np.arange(batch_size), labels] = 1
    grad_of_loss_wrt_pred_logits = (probabilities - one_hot_encoded_labels) / batch_size
    return grad_of_loss_wrt_pred_logits

def initialize_weights(input_size, output_size):
    # Kaiming-He Initialization
    fan_in = input_size
    uniform_dist_range = np.sqrt(6 / fan_in).astype(np.float32)
    weight = (np.random.rand(input_size, output_size).astype(np.float32) * 2 - 1) * uniform_dist_range
    return weight.astype(np.float32)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Parameters of first linear layer
        self.weight1 = initialize_weights(input_size, hidden_size)
        self.bias1 = np.zeros(hidden_size).astype(np.float32)

        # Parameters of second linear layer
        self.weight2 = initialize_weights(hidden_size, output_size)
        self.bias2 = np.zeros(output_size).astype(np.float32)
    
    def forward(self, input):
        # We need to cache the intermediate outputs (activations) required for backward pass
        fc1_out = linear(input, self.weight1, self.bias1)
        relu_out = relu(fc1_out)
        fc2_out = linear(relu_out, self.weight2, self.bias2)
        return input, fc1_out, relu_out, fc2_out
    
    def backward(self, grad_of_loss_wrt_fc2_out, cache):
        # Compute gradient of loss wrt every parameter of the model using chain rule
        input, fc1_out, relu_out, fc2_out = cache
        grad_of_loss_wrt_relu_out, grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias = \
            linear_derivative(relu_out, self.weight2, grad_of_loss_wrt_fc2_out)
        grad_of_loss_wrt_relu_in = relu_derivative(fc1_out, grad_of_loss_wrt_relu_out)
        grad_of_loss_wrt_input, grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias = \
            linear_derivative(input, self.weight1, grad_of_loss_wrt_relu_in)
        
        return (grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias,
            grad_of_loss_wrt_fc2_weight, grad_of_loss_wrt_fc2_bias)

    def update_parameters(self, gradients, learning_rate=10e-2):
        # Update parameters of the model using gradient of loss wrt parameters
        grad_of_loss_wrt_fc1_weight, grad_of_loss_wrt_fc1_bias, grad_of_loss_wrt_fc2_weight, \
        grad_of_loss_wrt_fc2_bias = gradients
        self.weight1 -= learning_rate * grad_of_loss_wrt_fc1_weight
        self.bias1 -= learning_rate * grad_of_loss_wrt_fc1_bias
        self.weight2 -= learning_rate * grad_of_loss_wrt_fc2_weight
        self.bias2 -= learning_rate * grad_of_loss_wrt_fc2_bias
        return

# Training Loop
def train_timed(model, x_train, y_train, epochs, batch_size, learning_rate):
    epoch_losses = []
    timing_stats = {'forward_pass': 0.0, 'backward_pass': 0.0, 'loss_computation': 0.0, 
                    'optimizer_step': 0.0, 'data_loading': 0.0}

    for epoch in range(epochs):
        iterations = x_train.shape[0] // batch_size
        epoch_loss = 0.0
        for iter in range(iterations):
            # Create data batch to be processed
            before_time = time.time()
            start_idx, end_idx = iter * batch_size, (iter + 1) * batch_size
            x, y = x_train[start_idx:end_idx], y_train[start_idx:end_idx]
            after_time = time.time()
            timing_stats['data_loading'] += after_time - before_time
            
            # Compute forward pass on the batch
            before_time = time.time()
            cache = model.forward(x)
            y_pred = cache[-1]
            after_time = time.time()
            timing_stats['forward_pass'] += after_time - before_time

            # Compute loss between the predicted and actual labels
            before_time = time.time()
            loss = cross_entropy_loss(y_pred, y)
            after_time = time.time()
            timing_stats['loss_computation'] += after_time - before_time
            epoch_loss += loss

            # Compute gradients for parameters during backward pass
            before_time = time.time()
            grad_of_loss_wrt_y_pred = cross_entropy_loss_derivative(y_pred, y)
            gradients = model.backward(grad_of_loss_wrt_y_pred, cache)
            after_time = time.time()
            timing_stats['backward_pass'] += after_time - before_time

            # Update parameters with gradients computed during backward pass
            before_time = time.time()
            model.update_parameters(gradients, learning_rate)
            after_time = time.time()
            timing_stats['optimizer_step'] += after_time - before_time

        print(f'Epoch {epoch}: Average Loss: {epoch_loss / iterations}')
        epoch_losses.append(epoch_loss / iterations)
    return epoch_losses, timing_stats


def main():
    TRAIN_SAMPLES, TEST_SAMPLES = 10000, 200
    EPOCHS = 10
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 8
    MNIST_DATASET_PATH = '../utils/data/'

    # Load data into gpus
    print('Loading MNIST dataset from binary files to nd-arrays')
    x_train, y_train, x_test, y_test = load_data_from_binary_files(MNIST_DATASET_PATH, TRAIN_SAMPLES, TEST_SAMPLES)
    print('Loading Complete')

    # Instantiate MLP model for classification of MNIST images
    print('Instantiating MLP Model')
    model = MLP(784, 256, 10)
    print('MLP Model Instantiation Complete')

    print('Starting Training')
    epoch_losses, timing_stats = train_timed(model, x_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    print('Training Completed')

    print_stats(timing_stats)


if __name__ == '__main__':
    np.random.seed(1)
    main()