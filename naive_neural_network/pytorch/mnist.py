import time
import math
import torch
import numpy as np
from torch import nn
from torch import optim

# DataLoading
def load_data_from_binary_files(dataset_dir, TRAIN_SIZE=10000, TEST_SIZE=200):
    x_train_path = f'{dataset_dir}/x_train.bin'
    y_train_path = f'{dataset_dir}/y_train.bin'
    x_test_path = f'{dataset_dir}/x_test.bin'
    y_test_path = f'{dataset_dir}/y_test.bin'

    x_train = np.fromfile(x_train_path, dtype=np.float32).reshape(60000, 784)
    y_train = np.fromfile(y_train_path, dtype=np.int32).reshape(60000)
    x_test = np.fromfile(x_test_path, dtype=np.float32).reshape(10000, 784)
    y_test = np.fromfile(y_test_path, dtype=np.int32).reshape(10000)

    x_train = torch.from_numpy(x_train)[:TRAIN_SIZE].reshape(-1, 1, 28, 28).to("cuda")
    y_train = torch.from_numpy(y_train)[:TRAIN_SIZE].reshape(-1).to(dtype=torch.long).to("cuda")
    x_test = torch.from_numpy(x_test)[:TEST_SIZE].reshape(-1, 1, 28, 28).to("cuda")
    y_test = torch.from_numpy(y_test)[:TEST_SIZE].reshape(-1).to(dtype=torch.long).to("cuda")

    return x_train, y_train, x_test, y_test

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)
    
    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def initialize_model(model):
    with torch.no_grad():
        # PyTorch stores weights as transposed matrices so that the elements in the columns of weight 
        # matrices are adjacent in memory. Other way to think of this is that the layout of weight 
        # matrix is column major and since pytorch is a row major library, it would interpret the column 
        # major weight matrix as transposed weight matrix in row major form

        # Kaiming-He Uniform Initialization
        # Prevents explosion of activation (and gradients) during the initial iterations
        # Required Distribution: mean=0, variance=(2/in_features)
        # For a Uniform Distribtuion U(-a, a), mean = 0, variance = a^2/3
        # So, a = sqrt(6 / in_features)
        fan_in_fc1 = model.fc1.weight.shape[1]
        uniform_dist_boundary = (6 / fan_in_fc1) ** 0.5
        model.fc1.weight.uniform_(-uniform_dist_boundary, uniform_dist_boundary)
        model.fc1.bias.zero_()

        fan_in_fc2 = model.fc2.weight.shape[1]
        uniform_dist_boundary = (6 / fan_in_fc2) ** 0.5
        model.fc2.weight.uniform_(-uniform_dist_boundary, uniform_dist_boundary)
        model.fc2.bias.zero_()
        return

def train_timed(model, optimizer, loss_fn, x_train, y_train, epochs, batch_size):
    epoch_losses = []
    timing_stats = {'forward_pass': 0.0, 'backward_pass': 0.0, 'loss_computation': 0.0, 
                    'optimizer_step': 0.0, 'data_loading': 0.0}

    model.train()
    for epoch in range(epochs):
        iterations = x_train.shape[0] // batch_size
        epoch_loss = 0.0
        for iter in range(iterations):
            # Create data batch to be processed
            before_time = time.time()
            start_idx, end_idx = iter * batch_size, (iter + 1) * batch_size
            x, y = x_train[start_idx:end_idx], y_train[start_idx:end_idx]
            torch.cuda.synchronize()
            after_time = time.time()
            timing_stats['data_loading'] += after_time - before_time
            
            # Compute forward pass on the batch
            before_time = time.time()
            y_pred = model(x)
            torch.cuda.synchronize()
            after_time = time.time()
            timing_stats['forward_pass'] += after_time - before_time

            # Compute loss between the predicted and actual labels
            before_time = time.time()
            loss = loss_fn(y_pred, y)
            torch.cuda.synchronize()
            after_time = time.time()
            timing_stats['loss_computation'] += after_time - before_time
            epoch_loss += loss.detach().item()

            # Set gradients to 0 so that they don't accumulate over the gradients of previous iteration
            optimizer.zero_grad()

            # Compute gradients for weights during backward pass
            before_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            after_time = time.time()
            timing_stats['backward_pass'] += after_time - before_time

            # Update weights with gradients computed during backward pass
            before_time = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            after_time = time.time()
            timing_stats['optimizer_step'] += after_time - before_time

        print(f'Epoch {epoch}: Average Loss: {epoch_loss / iterations}')
        epoch_losses.append(epoch_loss / iterations)
    return epoch_losses, timing_stats

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

def main():
    TRAIN_SAMPLES, TEST_SAMPLES = 10000, 200
    EPOCHS = 10
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 8
    MNIST_DATASET_PATH = '../utils/data/'

    # Load data into gpus
    print('Loading MNIST dataset from binary files to Tensors')
    x_train, y_train, x_test, y_test = load_data_from_binary_files(MNIST_DATASET_PATH, TRAIN_SAMPLES, TEST_SAMPLES)
    print('Loading Complete')

    # Instantiate MLP model for classification of MNIST images
    print('Instantiating MLP Model')
    model = MLP(784, 256, 10).to("cuda")
    print('Instantiation Complete. Initializing...')
    initialize_model(model)
    print('MLP Model Initialization Complete')

    # Instantiate Loss Function
    print('Instantiating Loss Function')
    loss_fn = nn.CrossEntropyLoss()
    print('Loss Function Instantiated')

    # Instantiate Optimizer
    print('Instantiating Optimizer')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print('Optimizer Instantiated')

    print('Starting Training')
    epoch_losses, timing_stats = train_timed(model, optimizer, loss_fn, x_train, y_train, EPOCHS, BATCH_SIZE)
    print('Training Completed')

    print_stats(timing_stats)


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.set_float32_matmul_precision("highest")
    main()