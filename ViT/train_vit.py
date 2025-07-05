# ViT/train_vit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====== Configuration ======
num_epochs = 10
lr = 2e-5
batch_size = 8
results_dir = "./ViT/results"
os.makedirs(results_dir, exist_ok=True)
model_name_suffix = f"{num_epochs}epochs"

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ====== Data ======
data_dir = "./dataset_11"
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
num_classes = len(classes)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ====== Model ======
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes
)
model.to(device)

# ====== Optimizer & Loss ======
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ====== Training ======
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(pixel_values=imgs).logits
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # ====== Validation ======
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(pixel_values=imgs).logits
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    avg_val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}")

# ====== Save model ======
model_path = os.path.join(results_dir, f"vit_model_{model_name_suffix}.pth")
torch.save(model.state_dict(), model_path)
print(f"[INFO] Model saved to {model_path}")

# ====== Plot Loss Curve ======
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.title("Loss Curve")
plt.savefig(os.path.join(results_dir, f"vit_loss_curve_{model_name_suffix}.png"))
plt.close()

# ====== Plot Accuracy Curve ======
plt.figure()
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.title("Accuracy Curve")
plt.savefig(os.path.join(results_dir, f"vit_accuracy_curve_{model_name_suffix}.png"))
plt.close()
