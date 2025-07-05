import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import json

# --- Configuration ---
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
LABEL_MAP = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}
LABEL_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load config for test set path ---
with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = json.load(f)

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test'))

# --- Model Setup ---
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Dataset Loader for Test Set ---
class TestMRIDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        for class_name, label in label_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(class_dir, fname))
                        self.labels.append(label)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate():
    # Load test dataset
    test_dataset = TestMRIDataset(test_dir, LABEL_MAP, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    # Save confusion matrix plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ResNet Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'resnet_confusion_matrix.png'))
    plt.close()

    # Print results only (no classification report)
    print("\nResNet Performance Evaluation")
    print("============================")
    print(f"Test Set Size: {len(test_dataset)}")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nConfusion matrix plot saved as 'resnet_confusion_matrix.png'.")

if __name__ == "__main__":
    evaluate() 