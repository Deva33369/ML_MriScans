# ViT/train_vit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Paths
data_dir = "./dataset_11"
results_dir = "./ViT/results"
os.makedirs(results_dir, exist_ok=True)

# Classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
num_classes = len(classes)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes
)
model.to(device)

# Optimizer & Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(pixel_values=imgs).logits
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(pixel_values=imgs).logits
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
    
    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save model
model_path = os.path.join(results_dir, "vit_model.pth")
torch.save(model.state_dict(), model_path)
print(f"[INFO] Model saved to {model_path}")

# Plot Loss curve
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_dir, "vit_loss_curve.png"))
plt.show()
