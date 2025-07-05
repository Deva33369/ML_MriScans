# ViT/test_vit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# ====== Configuration ======
model_name_suffix = "Best Model" 

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ====== Paths ======
data_dir = "./test"
results_dir = "./ViT/results"
# model_path = os.path.join(results_dir, f"vit_model_{model_name_suffix}.pth")
model_path = os.path.join(results_dir, "vit_model_best.pth")


# ====== Classes ======
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
num_classes = len(classes)

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Correct mean/std for ViT
                         std=[0.229, 0.224, 0.225])
])

# ====== Dataset ======
test_dataset = datasets.ImageFolder(root="./test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ====== Load Model ======
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ====== Evaluation ======
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(pixel_values=imgs).logits
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ====== Metrics ======
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"[INFO] Test Accuracy for {model_name_suffix}: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ====== Plot & Save Confusion Matrix ======
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.title(f"ViT Confusion Matrix ({model_name_suffix}, Acc={acc:.4f})")
plt.colorbar()
plt.xticks(ticks=np.arange(num_classes), labels=classes, rotation=45)
plt.yticks(ticks=np.arange(num_classes), labels=classes)

# Annotate values
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')

plt.tight_layout()
cm_path = os.path.join(results_dir, f"vit_confusion_matrix_{model_name_suffix}.png")
plt.savefig(cm_path)
plt.show()
print(f"[INFO] Confusion matrix saved to {cm_path}")
