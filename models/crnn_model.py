import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_channel=1, num_classes=37, hidden_size=256):
        super(CRNN, self).__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channel, 64, 3, 1, 1),   # (B, 1, H, W) → (B, 64, H, W)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                    # (B, 64, H/2, W/2)

            nn.Conv2d(64, 128, 3, 1, 1),           # (B, 64, H/2, W/2) → (B, 128, H/2, W/2)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)                     # (B, 128, H/4, W/4)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=128 * 8,                    # (C*H)
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully Connected
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # (B, W, hidden*2) → (B, W, num_classes)

    def forward(self, x):
        x = self.cnn(x)                            # (B, 128, H/4, W/4)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)                  # (B, W, C, H)
        x = x.view(b, w, c * h)                    # (B, W, C*H)
        x, _ = self.lstm(x)                        # (B, W, hidden*2)
        x = self.fc(x)                             # (B, W, num_classes)
        x = x.permute(1, 0, 2)                     # (W, B, num_classes)
        return x
