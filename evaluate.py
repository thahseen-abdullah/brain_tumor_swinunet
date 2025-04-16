# evaluate.py

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import get_loader
from swinunet_model import SwinUNet

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Get only validation loader from updated get_loader()
_, val_loader = get_loader("dataset", batch_size=8)

# Load trained model
model = SwinUNet().to(DEVICE)
model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=DEVICE))
model.eval()

# Evaluation loop
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        _, probs = model(imgs)
        preds = (probs > 0.5).float().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Results
print(classification_report(y_true, y_pred, target_names=["No‑Tumor", "Tumor"]))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
