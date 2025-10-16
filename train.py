import torch
import torch.nn as nn
import torch.optim as optim
from models.crnn_model import CRNN
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN(img_channel=1, num_classes=37).to(device)
loss_fn = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Dummy data
batch_size = 16
input_data = torch.randn(batch_size, 1, 32, 128).to(device)   # (B, C, H, W)
targets = torch.randint(1, 37, (batch_size * 5,), dtype=torch.long).to(device)
input_lengths = torch.full(size=(batch_size,), fill_value=32, dtype=torch.long)
target_lengths = torch.randint(5, 10, (batch_size,), dtype=torch.long)

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(input_data)                     # (W, B, num_classes)
    log_probs = F.log_softmax(outputs, dim=2)
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")
