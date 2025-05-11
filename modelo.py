# modelo.py
import torch.nn as nn

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
