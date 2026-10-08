import torch.nn as nn
from vocab import TOKENS


class CRNN(nn.Module):
    def __init__(self, num_classes: int = len(TOKENS)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 160x800 -> 80x400

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 40x200

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 20x100

            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # 10x100

            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # 5x100

            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((5, 1)),   # 1x100
        )

        self.sequence = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=False,
        )

        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        f = self.features(x)
        f = f.squeeze(2)
        f = f.permute(2, 0, 1)
        y, _ = self.sequence(f)
        y = self.head(y)
        return y
    