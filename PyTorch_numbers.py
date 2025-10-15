import ssl
import certifi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Eszköz kiválasztása (GPU ha elérhető)
device = torch.device("cpu")

# Adatok előkészítése
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_accuracies = []

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validáció
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    val_accuracy = correct / len(test_loader.dataset)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {val_accuracy:.4f}")

# Teszt képek
plt.figure(figsize=(12, 4))
for i in range(5):    
    image, label = mnist_test[i]    
    image_input = image.unsqueeze(0).to(device)
    model.eval()    
    with torch.no_grad():        
        output = model(image_input)        
        predicted_label = output.argmax(dim=1).item()    
    plt.subplot(1, 5, i + 1)    
    plt.imshow(image.squeeze(), cmap='gray')    
    plt.axis('off')    
    plt.title(f"Pred: {predicted_label}")
plt.suptitle("Első 5 tesztkép és predikció (PyTorch)")
plt.tight_layout()
plt.show()

# Tanulási görbék megjelenítése
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Tanító veszteség')
plt.plot(val_accuracies, label='Validációs pontosság')
plt.xlabel('Epoch')
plt.ylabel('Érték')
plt.title('Tanulási görbék (PyTorch)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()