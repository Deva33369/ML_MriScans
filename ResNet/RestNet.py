# MRI Classification using ResNet (Transfer Learning)

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import gradio as gr

# --- Configuration ---
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
LABEL_MAP = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}  # glioma, meningioma, pituitary, no tumor
LABEL_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Setup ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Load trained model if available
model_path = 'best_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from {model_path}")
else:
    print("No trained model found. Using pretrained ResNet weights.")
model.eval()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_mri(img):
    image = img.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return LABEL_NAMES[int(predicted.item())]

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="Brain MRI Tumor Classifier",
    description="Upload a brain MRI scan to detect Glioma, Meningioma, Pituitary Tumor or No Tumor.",
    
)

if __name__ == "__main__":
    iface.launch() 