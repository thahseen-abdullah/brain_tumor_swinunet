# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from swinunet_model import SwinUNet
from data_preprocessing import get_loader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Get train and validation loaders
train_loader, val_loader = get_loader("dataset", batch_size=8)

# Initialize model, loss, optimizer
model = SwinUNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')

# Training loop
for epoch in range(8):
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        seg, prob = model(imgs)

        loss = criterion(prob.squeeze(), labels)  # ✅ FIXED LOSS LINE
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            seg, prob = model(imgs)

            loss = criterion(prob.squeeze(), labels)
            val_loss += loss.item()

            preds = (prob > 0.5).float().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Epoch {epoch+1}/8 | train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f} | acc {acc:.3f} | prec {prec:.3f} | rec {rec:.3f} | f1 {f1:.3f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "saved_models/best_model.pth")

print("Training done. Best model saved to saved_models/best_model.pth")
