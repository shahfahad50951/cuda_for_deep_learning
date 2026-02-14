import torch
import os
from torchvision import datasets, transforms
import numpy as np

def download_and_save_mnist_dataset(dir='./data'):
    print('Starting MNIST data download and saving process')
    os.makedirs(dir, exist_ok=True)

    # Download MNIST dataset
    # It has total 70000 Images, 60000 train and 10000 test images
    # Each image is of shape 28 x 28
    # Note: Transforms are layzily applied when a data item is accessed. If we directly access the underlying tensors
    # like train_dataset.data, it would not be normalized by the transforms
    # ToTensor() - > Transforms the pixel values from [0-255] to [0.0-1.0] range
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=dir, train=False, download=True, transform=transform)
    print('Downloaded MNIST data. Saving...')

    # Instantiate dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, len(train_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, len(test_dataset))

    # Read full data using the dataloader so that the transforms are applied
    x_train_tensor, y_train_tensor = next(iter(train_dataloader))
    x_test_tensor, y_test_tensor = next(iter(test_dataloader))

    # Convert the inputs and labels to numpy arrays (float32)
    x_train_np, y_train_np = x_train_tensor.numpy().astype(np.float32), y_train_tensor.numpy().astype(np.int32)
    x_test_np, y_test_np = x_test_tensor.numpy().astype(np.float32), y_test_tensor.numpy().astype(np.int32)

    # The numpy function tofile() writes raw stream of bytes into the files
    # This is helpful because we can read this data from any language
    x_train_np.tofile(f'{dir}/x_train.bin')
    y_train_np.tofile(f'{dir}/y_train.bin')
    x_test_np.tofile(f'{dir}/x_test.bin')
    y_test_np.tofile(f'{dir}/y_test.bin')

    print('Saved MNIST data')

if __name__ == '__main__':
    torch.manual_seed(1)
    download_and_save_mnist_dataset('./data')