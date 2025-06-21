# MRI Classification using ResNet (Transfer Learning)

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import gradio as gr
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import models as kerasmodel


# --- Configuration ---
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
LABEL_MAP = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}  # glioma, meningioma, pituitary, no tumor
LABEL_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

LABEL_NAMES_CNN = ['glioma', 'meningioma', 'notumor', 'pituitary']        # For CNN
DISPLAY_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)

dataset_11 = config["dataset_11"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RsnNet Model Setup ---
model_ResNet = models.resnet18(pretrained=True)
model_ResNet.fc = nn.Linear(model_ResNet.fc.in_features, NUM_CLASSES)
model_ResNet = model_ResNet.to(device)

# Load trained model if available
model_ResNet_path = os.path.join(os.path.dirname(__file__), 'ResNet', 'best_model.pth')
if os.path.exists(model_ResNet_path):
    model_ResNet.load_state_dict(torch.load(model_ResNet_path, map_location=device))
    print(f"Loaded trained model from {model_ResNet_path}")
else:
    print("No trained model found. Using pretrained ResNet weights.")
model_ResNet.eval()

# --- Keras CNN Model Setup ---

def build_cnn_model():
    return tf.keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(224, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(224, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])

keras_CNN = build_cnn_model()
keras_CNN.load_weights(os.path.join(os.path.dirname(__file__), 'CNN', 'best_weights.weights.h5'))


def keras_preprocess(img):
    img = img.resize(IMAGE_SIZE)
    img = img.convert("RGB")
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_mri(img, model_choice):
    if model_choice == "ResNet":
        image = img.convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        selected_model = model_ResNet
        with torch.no_grad():
            output = selected_model(image)
            _, predicted = torch.max(output, 1)
        return LABEL_NAMES[int(predicted.item())]
    elif model_choice == "CNN":
        if keras_CNN is None:
            return "No trained Keras CNN model found."
        arr = keras_preprocess(img)
        preds = keras_CNN.predict(arr)
        class_idx = np.argmax(preds, axis=1)[0]
        return DISPLAY_LABELS[class_idx]
    elif model_choice == "SVM":
        return "SVM model not implemented"
    elif model_choice == "ViT":
        return "ViT model not implemented"
    else:
        return "Unknown model"

iface = gr.Interface(
    fn=predict_mri,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(choices=["ResNet", "CNN", "SVM", "ViT"], label="Select Model", value="ResNet")
    ],
    outputs=gr.Label(num_top_classes=1),
    title="Brain MRI Tumor Classifier",
    description="Upload a brain MRI scan and select a model to detect Glioma, Meningioma, Pituitary Tumor or No Tumor.",
    examples=[
        [os.path.join(dataset_11, "glioma", "Tr-gl_0023.jpg")],
        [os.path.join(dataset_11, "meningioma", "Tr-me_0014.jpg")],
        [os.path.join(dataset_11, "pituitary", "Tr-pi_0012.jpg")],
        [os.path.join(dataset_11, "notumor", "Tr-no_0016.jpg")]
    ]
)

if __name__ == "__main__":
    iface.launch() 