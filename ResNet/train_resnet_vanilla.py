import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import json
import random

# Configuration
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = json.load(f)

dataset_11 = config["dataset_11"]

LABEL_MAP = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}
LABEL_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# No data augmentation, just resize and normalize
default_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_dataset():
    image_paths = []
    labels = []
    for class_name in LABEL_MAP.keys():
        class_dir = os.path.join(dataset_11, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(LABEL_MAP[class_name])
    return image_paths, labels

def create_data_loaders():
    image_paths, labels = load_dataset()
    if len(image_paths) == 0:
        print("ERROR: No images found in dataset. Check dataset_11 path and class folders.")
        exit(1)
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    if len(combined) == 0:
        print("ERROR: No images to split after shuffling.")
        exit(1)
    image_paths, labels = zip(*combined)
    # Simple split (not stratified)
    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    train_labels = labels[:split_idx]
    val_labels = labels[split_idx:]
    if len(train_paths) == 0 or len(val_paths) == 0:
        print(f"ERROR: Not enough images to split. Train: {len(train_paths)}, Val: {len(val_paths)}")
        exit(1)
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    train_dataset = MRIDataset(train_paths, train_labels, default_transform)
    val_dataset = MRIDataset(val_paths, val_labels, default_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader

def create_model():
    model = models.resnet18(pretrained=True)
    # No layer freezing, all layers trainable
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)

def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
    # Save only the last model
    torch.save(model.state_dict(), 'resnet_vanilla_last_model.pth')
    print('Training complete. Last model saved as resnet_vanilla_last_model.pth')
    print(f'Final Train Acc: {train_accuracies[-1]:.2f}% | Final Val Acc: {val_accuracies[-1]:.2f}%')

def main():
    train_loader, val_loader = create_data_loaders()
    model = create_model()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main() 