import prepare_dataset
from prepare_dataset import get_data_loader
from model import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Subset
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, device, epochs=3):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), label)
            loss.backward()
            optimizer.step()

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
        return 100 * correct / total




labeled_pool = prepare_dataset.labeld_indices.tolist()
unlabeled_pool = prepare_dataset.unlabled_indices.tolist()

train_accs = []
test_accs = []


n_queries = 10               
query_batch_size = 500      
epochs_per_round = 3  

class WrappedModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = self.model(X_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def query_uncertainty_sampling(model, unlabeled_indices, n_instances):
    model.eval()
    subset = Subset(prepare_dataset.train_data, unlabeled_indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False)

    all_probs = []
    for data, _ in loader:
        data = data.to(device)
        with torch.inference_mode():
            outputs = model(data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    uncertainty = 1 - np.max(all_probs, axis=1)

    # Get top `n_instances` most uncertain **positions**
    selected_pos = np.argsort(uncertainty)[-n_instances:]

    # Convert positions back to dataset indices
    selected_indices = [unlabeled_indices[i] for i in selected_pos]
    return selected_indices


test_loader = prepare_dataset.test_loader

for round_num in range(n_queries):
    print(f"\n=== Round {round_num + 1}/{n_queries} ===")

    # Get current DataLoaders
    train_loader = get_data_loader(labeled_pool)
    
    # Initialize and train a new model from scratch
    model = SimpleCNN()
    train_model(model, train_loader, device, epochs=epochs_per_round)
    # Evaluate model
    train_acc = test_model(model, train_loader, device)
    test_acc = test_model(model, test_loader, device)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # If there are no more unlabeled samples to query, stop
    if len(unlabeled_pool) == 0:
        break

    selected_indices = query_uncertainty_sampling(model, unlabeled_pool, query_batch_size)


plt.plot(range(1, n_queries + 1), test_accs, label='Test Accuracy')
plt.plot(range(1, n_queries + 1), train_accs, label='Train Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Random Sampling Active Learning on CIFAR-10')
plt.legend()
plt.grid(True)
plt.show()