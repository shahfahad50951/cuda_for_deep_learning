import torch
import os
from torchvision import datasets, transforms
import numpy as np

def download_and_save_mnist_dataset(dir='./data'):
    os.makedirs(dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=dir, train=False, download=True, transform=transform)

    # Convert the inputs and labels to numpy arrays (float32)
    x_train, y_train = train_dataset.data.numpy().astype(np.float32), train_dataset.targets.numpy().astype(np.float32)
    x_test, y_test = test_dataset.data.numpy().astype(np.float32), test_dataset.targets.numpy().astype(np.float32)

    # The numpy function tofile() writes raw stream of bytes into the files
    # This is helpful because we can read this data from any language
    x_train.tofile(f'{dir}/x_train.bin')
    y_train.tofile(f'{dir}/y_train.bin')
    x_test.tofile(f'{dir}/x_test.bin')
    y_test.tofile(f'{dir}/y_test.bin')

if __name__ == '__main__':
    torch.manual_seed(1)
    download_and_save_mnist_dataset('./data')