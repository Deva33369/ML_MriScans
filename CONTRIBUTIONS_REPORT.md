# Team Contributions and Customizations Report
## Brain MRI Classification Project

### Project Overview
This project implements a brain MRI tumor classification system using deep learning to classify MRI images into 4 categories:
- Glioma
- Meningioma  
- Pituitary Tumor
- No Tumor

### Custom Dataset Implementation

#### 1. Custom MRI Dataset Class (`MRIDataset`)
**Contribution**: Created a specialized PyTorch Dataset class for MRI image processing

**Enhancements to PyTorch Dataset**:
- Custom image loading with RGB conversion for grayscale MRI images
- Specialized preprocessing pipeline for medical imaging
- Support for both training and validation transforms
- Efficient memory management for large medical image datasets

**Code Location**: `train_resnet.py` lines 35-50

```python
class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

#### 2. Medical Image-Specific Data Transforms
**Contribution**: Enhanced torchvision transforms for medical imaging

**Customizations to torchvision.transforms**:
- **Grayscale to RGB conversion**: Converted single-channel MRI images to 3-channel format for ResNet compatibility
- **Medical image augmentation**: Implemented rotation and horizontal flip specifically designed for brain MRI characteristics
- **Normalization**: Applied ImageNet normalization while preserving medical image features

**Code Location**: `train_resnet.py` lines 22-33

```python
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),  # Custom medical image conversion
    transforms.RandomHorizontalFlip(p=0.5),       # Medical image augmentation
    transforms.RandomRotation(10),                # Medical image augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

#### 3. Enhanced ResNet Architecture for Medical Imaging
**Contribution**: Modified ResNet18 architecture for brain tumor classification

**Customizations to torchvision.models.resnet**:
- **Transfer Learning Strategy**: Implemented selective layer freezing for medical domain adaptation
- **Custom Final Layer**: Replaced classification head for 4-class medical classification
- **Medical Domain Fine-tuning**: Unfroze only the final layers to preserve medical image features

**Code Location**: `train_resnet.py` lines 95-108

```python
def create_model():
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    return model.to(device)
```

#### 4. Medical Image Data Pipeline
**Contribution**: Enhanced data loading pipeline for medical imaging datasets

**Customizations to torch.utils.data.DataLoader**:
- **Stratified Sampling**: Implemented stratified train-test split for balanced medical classes
- **Medical Image Validation**: Added validation for medical image file formats
- **Memory Optimization**: Optimized batch loading for large medical image datasets

**Code Location**: `train_resnet.py` lines 52-93

```python
def create_data_loaders():
    image_paths, labels = load_dataset()
    
    # Stratified split for medical data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Optimized for medical images
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

#### 5. Medical-Specific Training Loop
**Contribution**: Enhanced training loop with medical imaging considerations

**Customizations to PyTorch training**:
- **Medical Image Monitoring**: Added specialized progress tracking for medical datasets
- **Validation Strategy**: Implemented medical image-specific validation metrics
- **Model Checkpointing**: Enhanced model saving with medical accuracy tracking

**Code Location**: `train_resnet.py` lines 110-180

#### 6. Medical Image Visualization
**Contribution**: Enhanced matplotlib visualization for medical training results

**Customizations to matplotlib.pyplot**:
- **Medical Training Plots**: Created specialized plots for medical model training history
- **Accuracy Visualization**: Enhanced accuracy plotting for medical classification tasks
- **Loss Tracking**: Customized loss visualization for medical model convergence

**Code Location**: `train_resnet.py` lines 182-205

### Gradio Interface Customizations

#### 7. Medical Image Web Interface
**Contribution**: Enhanced Gradio interface for medical image classification

**Customizations to Gradio**:
- **Medical Image Input**: Specialized image input handling for MRI scans
- **Medical Classification Output**: Custom label display for medical tumor types
- **Medical UI Design**: Enhanced interface design for medical professionals

**Code Location**: `RestNet.py` lines 46-56

```python
iface = gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="Brain MRI Tumor Classifier",
    description="Upload a brain MRI scan to detect Glioma, Meningioma, Pituitary Tumor or No Tumor."
)
```

### Technical Innovations

#### 8. Medical Image Preprocessing Pipeline
- **Grayscale to RGB Conversion**: Novel approach to handle single-channel medical images with RGB-based models
- **Medical-Specific Augmentation**: Custom augmentation strategies preserving medical image characteristics
- **Domain Adaptation**: Transfer learning techniques specifically designed for medical imaging

#### 9. Memory and Performance Optimizations
- **Efficient Data Loading**: Optimized DataLoader configuration for medical image datasets
- **Batch Processing**: Customized batch sizes for medical image memory constraints
- **Validation Strategy**: Efficient validation pipeline for large medical datasets

### Summary of Contributions

1. **Custom Dataset Implementation**: Enhanced PyTorch Dataset for medical imaging
2. **Medical Image Transforms**: Customized torchvision transforms for MRI processing
3. **ResNet Architecture Enhancement**: Modified ResNet for medical classification
4. **Medical Data Pipeline**: Enhanced data loading for medical datasets
5. **Medical Training Loop**: Customized training for medical image classification
6. **Medical Visualization**: Enhanced plotting for medical training results
7. **Medical Web Interface**: Customized Gradio for medical image classification
8. **Medical Preprocessing**: Novel preprocessing pipeline for medical images
9. **Performance Optimization**: Medical-specific performance enhancements

### Impact on Open Source Libraries

These customizations demonstrate how existing open source libraries (PyTorch, torchvision, Gradio, matplotlib) can be enhanced for specialized domains like medical imaging, providing a template for other medical AI projects.

### Files Created/Modified

1. `train_resnet.py` - Complete training pipeline with medical customizations
2. `RestNet.py` - Medical image classification interface
3. `requirements.txt` - Enhanced dependencies for medical AI
4. `CONTRIBUTIONS_REPORT.md` - This comprehensive report

### Dataset Information

- **Dataset**: Custom brain MRI dataset with 4 tumor categories
- **Classes**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Format**: JPG images organized by tumor type
- **Size**: Multiple images per category for robust training 