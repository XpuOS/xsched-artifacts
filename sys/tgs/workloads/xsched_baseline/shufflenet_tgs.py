# train a shufflenet on cifar10

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import shufflenet_v2_x1_0
import time
from pathlib import Path
import sys
import queue

# Define hyperparameters
batch_size = 16
num_epochs = 10000 # never stop
learning_rate = 0.01

latency_queue = queue.Queue()
for i in range(100):
    latency_queue.put(1)

# Set device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
else:
    device = torch.device("cpu")
    stream = None

print(f"Using device: {device}, stream: {hex(stream.cuda_stream)}")

# Modify dataset loading to use basic transforms
transform_basic = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset with minimal transforms
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_basic)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_basic)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define the full transforms to be applied per batch
def apply_transforms(images, is_training=True):
    if is_training:
        transforms_list = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transforms_list = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transforms_list(images)

# Load ShuffleNet V2 model without pre-trained weights
model = shufflenet_v2_x1_0(pretrained=False)

# Modify the final fully connected layer for CIFAR-10 (10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Move the model to the device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        begin = time.time()
        
        inputs = apply_transforms(inputs, is_training=True)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Evaluate
        batch_loss = running_loss / (i + 1)
        batch_acc = 100. * correct / total
        end = time.time()

        latency_queue.put(end - begin)
        latency_queue.get()

        sum = 0
        for item in latency_queue.queue:
            sum += item
        avg = sum / latency_queue.qsize()

        current_time = time.strftime("%H:%M:%S", time.localtime()) + f".{(time.time() % 1)*1000000:06.0f}"
        print(f"{{\"time\": \"{current_time}\", "
              f"\"batch\": \"{i+1}/{len(trainloader)}\", "
              f"\"time (ms)\": {(end - begin)*1000:.3f}, "
              f"\"thpt (iter/s)\": {batch_size / avg:.3f}, "
              f"\"loss\": {batch_loss:.4f}, "
              f"\"acc (%)\": {batch_acc:.2f}"
              f"}},", flush=True)
        
        # Evaluate on test set
        # test_loss, test_acc = evaluate()
        # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Evaluation loop
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            # Apply transforms here
            inputs = apply_transforms(inputs, is_training=False)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


# Main training loop
with torch.cuda.stream(stream):
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train()
        test_loss, test_acc = evaluate()
        
        # Step the scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        current_time = time.strftime("%H:%M:%S", time.localtime()) + f".{(time.time() % 1)*1000000:06.0f}"

        print(f"{{\"time\": \"{current_time}\", "
              f"\"epoch\": \"{epoch+1}/{num_epochs}\", "
              f"\"time (s)\": {epoch_time:.4f}, "
              f"\"train_loss\": {train_loss:.4f}, "
              f"\"train_acc (%)\": {train_acc:.2f}, "
              f"\"test_loss\": {test_loss:.4f}, "
              f"\"test_acc (%)\": {test_acc:.2f}"
              f"}},", flush=True)

print("Training finished!")

# Save the trained model
torch.save(model.state_dict(), "shufflenet_cifar10.pth")
print("Model saved as shufflenet_cifar10.pth")
