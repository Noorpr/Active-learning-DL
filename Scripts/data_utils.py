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


def get_dataloader(train_data, indices, batch_size=64, shuffle=True):
    subset = Subset(train_data, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)