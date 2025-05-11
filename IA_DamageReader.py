from PIL import Image
import torchvision.transforms as transforms
import os
from modelo import CarClassifier
import torch

# Transformação igual ao treino
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Caminho da pasta de teste
test_path = "testes"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelo
model = CarClassifier()
model.load_state_dict(torch.load("modelo_carro.pth"))
model.to(device)
model.eval()

# Rodando predições
for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, 1).item()

    label = "danificado" if pred == 0 else "normal"
    print(f"{img_name}: {label}")
