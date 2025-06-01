import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class CarClassifier(nn.Module):
    def __init__(self):
        super(CarClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64), nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: normal e danificado
        )

    def forward(self, x):
        return self.model(x)

carrosTreino = "carros"

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset
dataset = datasets.ImageFolder(carrosTreino, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega modelo pré-treinado e adapta
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Duas classes

model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Salvando modelo
torch.save(model.state_dict(), "modelo_carro.pth")

carrosTeste = "testes"

# Recarrega modelo treinado
model.load_state_dict(torch.load("modelo_carro.pth"))
model.eval()

for img_name in os.listdir(carrosTeste):
    img_path = os.path.join(carrosTeste, img_name)
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, 1).item()
        label = "danificado" if pred == 0 else "normal"
            
    # Mostrar imagem com título
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"{label}", fontsize=12)
    plt.show()
