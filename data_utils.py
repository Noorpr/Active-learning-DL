from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    return train_data, test_data

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=False, transform=transform)
    return train_data, test_data

def get_dataloader(train_data, indices, batch_size=64, shuffle=True):
    subset = Subset(train_data, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)