import torch
from models.crnn_model import CRNN
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN(img_channel=1, num_classes=37).to(device)
model.load_state_dict(torch.load("crnn_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    dummy_input = torch.randn(1, 1, 32, 128).to(device)
    output = model(dummy_input)
    probs = F.softmax(output, dim=2)
    pred = torch.argmax(probs, dim=2)
    print("Predicted sequence (index):", pred[:, 0].cpu().numpy())
