# overlay.py
"""
Generates a segmentation-based heatmap overlay from Swin‑UNet
and saves it with a color bar. Output: seg_overlay_vivid.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as T
from PIL import Image
from swinunet_model import SwinUNet

# ------------------ 1. Load Model ------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SwinUNet().to(DEVICE)
model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=DEVICE))
model.eval()

# ------------------ 2. Load Image ------------------
IMG_PATH = "dataset/yes/y12.jpg"  # <-- Change to any sample

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
img_tensor = transform(Image.open(IMG_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

# ------------------ 3. Segmentation Output ---------
seg_map, _ = model(img_tensor)  # (B, 1, H, W)
seg_map = torch.nn.functional.interpolate(seg_map, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
seg_map = seg_map.detach().cpu()
seg_map = seg_map / seg_map.max()  # normalize

# Optional: zero out weak activations
seg_map = seg_map * (seg_map > 0.3).float()

# ------------------ 4. Blend with Image ------------
# Original image for overlay
orig = (img_tensor.cpu()[0] * 0.5) + 0.5  # de-normalize
orig_np = orig.permute(1, 2, 0).numpy()

# Apply colormap to segmentation
seg_np = seg_map.numpy()
cmap = matplotlib.colormaps.get_cmap("plasma")  # updated for matplotlib 3.7+
heat_np = cmap(seg_np)[..., :3]  # discard alpha

# Blend heatmap and original
overlay_np = 0.4 * orig_np + 0.6 * heat_np
overlay_np = np.clip(overlay_np, 0, 1)

# ------------------ 5. Save Figure -----------------
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(overlay_np)
ax.axis("off")

# Add color bar legend
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Tumour Probability (0–1)")

plt.tight_layout()
plt.savefig("seg_overlay_vivid.png", dpi=300)
plt.close()
print("Saved overlay with colour‑bar → seg_overlay_vivid.png")
