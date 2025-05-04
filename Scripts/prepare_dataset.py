import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np


np.random.seed(13)
torch.manual_seed(13)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])



train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# print(train_data.data.shape)
# print(test_data.data.shape)

indices = np.arange(len(train_data))
np.random.shuffle(indices)

initial_labled_size = 1000
labeld_indices = indices[:initial_labled_size]
unlabled_indices = indices[:initial_labled_size:]

def get_data_loader(indicies, batch_size=64, shuffle=True):
    subset = Subset(train_data, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


test_loader = DataLoader(test_data, batch_size=32, shuffle=False)