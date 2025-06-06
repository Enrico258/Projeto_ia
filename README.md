Enrico Cuono Alves Pereira - 10402875

# Resumo da proposta:

Durante o uso cotidiano de veículos, é comum que ocorram arranhões e amassados, que algumas vezes, podem passar despercebidos. Pensando nisso, surge a proposta de desenvolver uma inteligência artificial capaz de identificar automaticamente danos na lataria de um carro, como riscos e amassados. Essa tecnologia pode ser amplamente aplicada por proprietários de veículos, concessionárias e seguradoras, oferecendo agilidade, praticidade e precisão na detecção de danos, além de facilitar processos de avaliação, manutenção e seguros. Podem ser encontradas maiores dificuldades na identificação de imagens com iluminações fortes e designs de carrocerias que possam parecer danos, como curvas acentuadas.

O trai será treinada utilizando PyTorch, com 70 imagens de carros danificados adquiridas no Kaggle (https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset) e não danificados da cars.com.
Será utilizado o python, sendo a bilbioteca do pyTorch, para o treinamento da rede neural e o google colab para ser utilizado como ambiente.

O modelo será capaz de classificar imagens de carros em duas categorias principais:
- Carro normal (sem danos visíveis)
- Carro danificado (com amassados, arranhões ou outros tipos de danos)

# Referências: 
- https://www.tensorflow.org/tutorials/quickstart/beginner?hl=pt-br
- https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment

# Entregas: 
- [Código completo e executável em um notebook Python](notebookFinalProjeto.ipynb)
- [Aplicação streamlit](#Aplicação-Streamlit)
- Link do vídeo: https://youtu.be/i2FiMoWvNNs 

##  Aplicação Streamlit
Para consumit o modelo via uma aplicação Streamlit:

### 1. Clone o repositório

```bash
git clone https://github.com/Enrico258/Projeto_ia.git
```

### 2. Instale as dependências

```bash
pip install streamlit gdown
```

### 3. Execute a aplicação Streamlit

```bash
cd .\Projeto_ia\
streamlit run streamlitApp.py
```

### 4. Acesse a aplicação

Após executar o comando acima, o Streamlit abrirá automaticamente seu navegador. Caso isso não aconteça, acesse manualmente:

```
http://localhost:8501
```
