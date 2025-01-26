# src/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from src.model import SimpleCNN

def train_model(epochs=10, batch_size=64, learning_rate=0.001):
    device = torch.device("ipu" if hasattr(torch, "has_ipu") and torch.has_ipu else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
    
    torch.save(model.state_dict(), 'models/simple_cnn.pth')

if __name__ == "__main__":
    train_model()