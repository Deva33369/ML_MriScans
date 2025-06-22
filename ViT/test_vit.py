# ViT/test_vit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Paths
data_dir = "./dataset_11"
results_dir = "./ViT/results"
model_path = os.path.join(results_dir, "vit_model.pth")

# Classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
num_classes = len(classes)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Evaluate
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(pixel_values=imgs).logits
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"[INFO] Test Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# Save Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.title(f"ViT Confusion Matrix (Acc={acc:.4f})")
plt.colorbar()
plt.xticks(ticks=np.arange(num_classes), labels=classes, rotation=45)
plt.yticks(ticks=np.arange(num_classes), labels=classes)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "vit_confusion_matrix.png"))
plt.show()
