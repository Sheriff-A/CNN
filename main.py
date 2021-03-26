import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
from pathlib import Path

from time import time
from datetime import datetime

from model import CNN


# Utility Methods
def print_epoch_result(result):
    print('Epoch Summary')
    print('Epoch: {}'.format(result[0]))
    print('Loss: {:.4f}'.format(result[1]))
    print('Number Correctly Classified: {}'.format(result[2]))
    print('Accuracy: {:.2f}%'.format(result[3]))
    print()


def make_dir(path):
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)


# Paths
PROJ_DIR = Path('.')
DATA_DIR = PROJ_DIR / 'DATA'
DATE = datetime.now().strftime('%Y-%b-%d@%H;%M;%S')
MODEL_DIR = PROJ_DIR / 'MODEL' / f'{DATE}'
MODEL_NAME = 'cnn_model.pt'

# Set this to True if the data has not been downloaded
download = False

# Model Parameters
in_channels = 3
output = 10

# Training Parameters
lr = 0.001
epochs = 10
batch_size = 100
shuffle = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform The Data
transform = transforms.ToTensor()

print('----- LOADING CIFAR10 DATASET -----')

# Load the Data
make_dir(DATA_DIR)
train_data = datasets.CIFAR10(root=DATA_DIR, download=download, train=True, transform=transform)
test_data = datasets.CIFAR10(root=DATA_DIR, download=download, train=False, transform=transform)

# Prepare the Data Loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

# Size of Data
training_size = len(train_loader) * train_loader.batch_size
testing_size = len(test_loader) * test_loader.batch_size
print("Training Size:", training_size)
print("Testing Size:", testing_size)
print()

print('----- MODEL -----')
model = CNN(in_channels=in_channels, output=output)
print(model)
print()

print('----- TRAINING PARAMETERS -----')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
print('Optimizers and Criterion Setup')
print(f'Learning Rate: {lr}, Epochs: {epochs}, Shuffle: {shuffle}, Device: {device}')
print()

print('----- TRAINING -----')
# Summary of Training Results for Graphing
# [epoch, loss, number correctly classified, accuracy]
training_results = []

# Send Model and Loss Function to Device (CPU or GPU if available)
model.to(device)
criterion.to(device)

start_train_time = time()
for e in range(epochs):
    train_corr = 0

    # Run the training in batches
    for idx, (data, label) in enumerate(train_loader):
        idx += 1
        # Send Data and Labels to Device (CPU or GPU if available)
        data = data.to(device)
        label = label.to(device)

        # Apply Model
        output = model(data)
        loss = criterion(output, label)

        # Tally the number of correct predictions
        predicted = torch.max(output.data, 1)[1]
        batch_corr = (predicted == label).sum()
        train_corr += batch_corr

        # Update Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print Interim Results
        if idx % 100 == 0:
            print('Epoch: {epc}, Loss: {l_item:.4f}, Accuracy: {acc:.2f}%'.format(
                epc=e,
                l_item=loss.item(),
                acc=train_corr.item() * 100 / (100 * idx)
            ))

    training_results.append([e, loss.item(), train_corr.item(), train_corr.item() * 100 / (100 * idx)])
    train_duration = time() - start_train_time
    print('Training Duration:', train_duration, 's')
    print()

print('Final Training Accuracy: {:.2f}'.format(training_results[len(training_results) - 1][3]))

print('----- TESTING -----')
test_corr = 0
start_test_time = time()
with torch.no_grad():
    for _, (data, label) in enumerate(test_loader):
        # Send Data to Device
        data = data.to(device)
        label = label.to(device)

        # Apply model
        output = model(data)
        predicted = torch.max(output.data, 1)[1]
        batch_corr = (predicted == label).sum()
        test_corr += batch_corr

test_duration = time() - start_test_time
print('Testing Duration:', test_duration, 's')
print("Accuracy: {:.2%}".format(test_corr.item() / testing_size))
print()

print('----- SAVING MODEL -----')
make_dir(MODEL_DIR)
torch.save(model.state_dict(), MODEL_DIR / MODEL_NAME)
print('Path:', MODEL_DIR / MODEL_NAME)
print("DONE")
