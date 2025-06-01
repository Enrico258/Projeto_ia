import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import gdown

if not os.path.exists("modelo_carro.pth"):
    # Download do arquivo pth caso não haja
    url = f"https://drive.google.com/uc?id=1SuawSetbel0spkjwAKUH9LJpkbfyMwKn"
    output = "modelo_carro.pth"
    gdown.download(url, output, quiet=False)

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforma imagem (mesmo usado no treino!)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ajuste conforme o tamanho usado no treino
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carrega o modelo
@st.cache_resource
def carregar_modelo():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("modelo_carro.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Função de predição
def prever_imagem(image, model):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()
    return "danificado" if pred == 0 else "normal"

# Interface Streamlit
st.title("Classificador de Carros - Danificado ou Normal")
st.write("Faça upload de uma imagem de carro para verificar sua condição.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

    model = carregar_modelo()
    label = prever_imagem(image, model)

    st.markdown(f"### Resultado: **{label.upper()}** 🚗")

