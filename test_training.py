#!/usr/bin/env python3
"""
Test script to validate the brain MRI classification training pipeline
and demonstrate the customizations to open source libraries.
"""

import os
import sys
from train_resnet import load_dataset, create_data_loaders, create_model

def test_dataset_loading():
    """Test the custom dataset loading functionality"""
    print("Testing custom dataset loading...")
    
    try:
        image_paths, labels = load_dataset()
        print(f"✓ Successfully loaded {len(image_paths)} images")
        print(f"✓ Found {len(set(labels))} unique classes")
        
        # Count images per class
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("Images per class:")
        for label, count in class_counts.items():
            print(f"  Class {label}: {count} images")
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

def test_data_loaders():
    """Test the custom data loader creation"""
    print("\nTesting custom data loaders...")
    
    try:
        train_loader, val_loader = create_data_loaders()
        print(f"✓ Training loader: {len(train_loader)} batches")
        print(f"✓ Validation loader: {len(val_loader)} batches")
        
        # Test a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"✓ Batch {batch_idx}: {images.shape}, labels: {labels.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Error creating data loaders: {e}")
        return False

def test_model_creation():
    """Test the custom ResNet model creation"""
    print("\nTesting custom ResNet model...")
    
    try:
        model = create_model()
        print(f"✓ Model created successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def test_customizations():
    """Test all customizations to open source libraries"""
    print("=" * 60)
    print("TESTING CUSTOMIZATIONS TO OPEN SOURCE LIBRARIES")
    print("=" * 60)
    
    tests = [
        ("Custom Dataset Loading", test_dataset_loading),
        ("Custom Data Loaders", test_data_loaders),
        ("Custom ResNet Model", test_model_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All customizations working correctly!")
        print("\nReady to run full training with:")
        print("python train_resnet.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = test_customizations()
    sys.exit(0 if success else 1) 