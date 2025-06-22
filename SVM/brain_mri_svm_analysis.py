import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import hashlib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class BrainMRILoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
    def load_images(self, img_size=(128, 128)):
        images = []
        labels = []
        
        print("Loading brain MRI images...")
        
        for class_id, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.data_path, class_name)
            
            if os.path.exists(class_folder):
                image_files = [f for f in os.listdir(class_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                print(f"Loading {len(image_files)} {class_name} images...")
                
                for img_file in image_files:
                    img_path = os.path.join(class_folder, img_file)
                    
                    # Read image in grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize image
                        img_resized = cv2.resize(img, img_size)
                        
                        # Normalize to 0-1 range
                        img_normalized = img_resized / 255.0
                        
                        images.append(img_normalized)
                        labels.append(class_id)
        
        print(f"Total images loaded: {len(images)}")
        return np.array(images), np.array(labels)

class FeatureExtractor:
    
    def extract_statistical_features(self, image):
        features = []
        
        # Flatten image
        pixels = image.flatten()
        
        # Basic statistics
        features.append(np.mean(pixels))      
        features.append(np.std(pixels))       
        features.append(np.min(pixels))       
        features.append(np.max(pixels))       
        features.append(np.median(pixels))    
        
        # Percentiles
        features.append(np.percentile(pixels, 25))  
        features.append(np.percentile(pixels, 75))  
        
        return features
    
    def extract_texture_features(self, image):
        from skimage.feature import local_binary_pattern
        img_uint8 = (image * 255).astype(np.uint8)
        radius = 1
        n_points = 8
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                              range=(0, n_points + 2), density=True)
        
        return hist.tolist()
    
    def extract_edge_features(self, image):
        img_uint8 = (image * 255).astype(np.uint8)

        # Calculate gradients
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
        # Edge features
        features = []
        features.append(np.mean(magnitude))    
        features.append(np.std(magnitude))     
        features.append(np.max(magnitude))     
        
        # Edge density 
        edges = cv2.Canny(img_uint8, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return features
    
    def extract_shape_features(self, image):
        img_uint8 = (image * 255).astype(np.uint8)
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour area
            area = cv2.contourArea(largest_contour)
            features.append(area)
            
            # Contour perimeter
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter)
            
            # Aspect ratio of bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            features.append(aspect_ratio)
            
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def extract_all_features(self, image):
        all_features = []
        
        # Statistical features
        stat_features = self.extract_statistical_features(image)
        all_features.extend(stat_features)
        
        # Texture features
        texture_features = self.extract_texture_features(image)
        all_features.extend(texture_features)
        
        # Edge features
        edge_features = self.extract_edge_features(image)
        all_features.extend(edge_features)
        
        # Shape features
        shape_features = self.extract_shape_features(image)
        all_features.extend(shape_features)
        
        return np.array(all_features)

class EnhancedSVMClassifier:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = self.get_feature_names()
    
    def get_feature_names(self):
        names = []
        names.extend(['mean', 'std', 'min', 'max', 'median', 'q25', 'q75'])
        names.extend([f'lbp_bin_{i}' for i in range(10)])  # 10 LBP bins
        names.extend(['edge_mean', 'edge_std', 'edge_max', 'edge_density'])
        names.extend(['contour_area', 'contour_perimeter', 'aspect_ratio'])
        
        return names
    
    def extract_features_from_images(self, images):
        print("Extracting features from images...")
        
        all_features = []
        for i, image in enumerate(images):
            if i % 50 == 0:
                print(f"Processed {i}/{len(images)} images")
            
            features = self.feature_extractor.extract_all_features(image)
            all_features.append(features)
        
        features_array = np.array(all_features)
        print(f"Extracted {features_array.shape[1]} features per image")
        
        return features_array
    
    def train_and_compare_models(self, X_train, y_train, X_test, y_test):
        print("\nTraining and comparing SVM models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define different SVM configurations to test
        models = {
            'Linear SVM': SVC(kernel='linear', random_state=42),
            'RBF SVM': SVC(kernel='rbf', random_state=42),
            'Polynomial SVM': SVC(kernel='poly', degree=3, random_state=42)
        }
        
        results = {}
        
        # Train each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        
        return results, X_train_scaled, X_test_scaled
    
    def optimize_best_model(self, X_train, y_train):
        print("\nOptimizing hyperparameters...")
        
        # Define parameter grid for RBF kernel 
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        # Grid search
        svm = SVC(kernel='rbf', random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        return grid_search

class ResultsExporter:
    def __init__(self, class_names):
        self.class_names = class_names
        self.results_folder = "results"
        self.setup_results_folder()
        
    def setup_results_folder(self):
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print(f"📁 Created results folder: {self.results_folder}/")
    
    def generate_result_hash(self, results, y_pred):
        data_str = str(sorted([(k, v['accuracy']) for k, v in results.items()])) + str(y_pred.tolist())
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def should_export(self, filename, result_hash):
        filepath = os.path.join(self.results_folder, filename)
        hash_file = os.path.join(self.results_folder, f".{filename}.hash")
        
        if not os.path.exists(filepath):
            return True
            
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            return stored_hash != result_hash
        
        return True
    
    def save_hash(self, filename, result_hash):
        hash_file = os.path.join(self.results_folder, f".{filename}.hash")
        with open(hash_file, 'w') as f:
            f.write(result_hash)
    
    def plot_class_distribution(self, labels, result_hash):
        filename = "1_class_distribution.png"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        plt.figure(figsize=(12, 5))
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.subplot(1, 2, 1)
        bars = plt.bar([self.class_names[i] for i in unique], counts, 
                      color=['red', 'blue', 'green', 'orange'])
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Images', fontsize=12)
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=[self.class_names[i] for i in unique], 
               autopct='%1.1f%%', colors=['red', 'blue', 'green', 'orange'],
               startangle=90)
        plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.save_hash(filename, result_hash)
        print(f"💾 Saved: {filename}")
        plt.show()
    
    def plot_sample_images(self, images, labels, result_hash, samples_per_class=3):
        filename = "2_sample_images.png"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        fig, axes = plt.subplots(len(self.class_names), samples_per_class, figsize=(12, 10))
        
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = labels == class_idx
            class_images = images[class_mask]
            
            if len(class_images) >= samples_per_class:
                np.random.seed(42)
                sample_indices = np.random.choice(len(class_images), samples_per_class, replace=False)
                
                for i, sample_idx in enumerate(sample_indices):
                    axes[class_idx, i].imshow(class_images[sample_idx], cmap='gray')
                    axes[class_idx, i].set_title(f'{class_name}', fontsize=12, fontweight='bold')
                    axes[class_idx, i].axis('off')
        
        plt.suptitle('Sample Brain MRI Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.results_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.save_hash(filename, result_hash)
        print(f"💾 Saved: {filename}")
        plt.show()
    
    def plot_model_comparison(self, results, result_hash):
        filename = "3_model_comparison.png"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        plt.figure(figsize=(12, 6))
        
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('SVM Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.results_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.save_hash(filename, result_hash)
        print(f"💾 Saved: {filename}")
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, result_hash):
        filename = "4_confusion_matrix.png"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(self.results_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.save_hash(filename, result_hash)
        print(f"💾 Saved: {filename}")
        plt.show()
    
    def plot_feature_importance(self, classifier, result_hash):
        filename = "5_feature_importance.png"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        if hasattr(classifier.best_model, 'coef_') and classifier.best_model.coef_ is not None:
            plt.figure(figsize=(12, 8))
            
            coef = np.abs(classifier.best_model.coef_[0])
            feature_names = classifier.feature_names
            
            top_indices = np.argsort(coef)[-15:]
            top_features = [feature_names[i] for i in top_indices]
            top_values = coef[top_indices]
            
            plt.barh(range(len(top_values)), top_values, color='steelblue')
            plt.yticks(range(len(top_values)), top_features)
            plt.xlabel('Feature Importance (Absolute Coefficient)', fontsize=12)
            plt.title('Top 15 Most Important Features (Linear SVM)', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(self.results_folder, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.save_hash(filename, result_hash)
            print(f"💾 Saved: {filename}")
            plt.show()
        else:
            print("⚠️ Feature importance not available for non-linear kernels")
    
    def create_results_summary(self, results, y_true, y_pred, classifier, result_hash):
        filename = "6_comprehensive_summary.png"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model Comparison
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        bars = axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Model Performance Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        for bar, accuracy in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        im = axes[0, 1].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0, 1].set_title('Confusion Matrix', fontweight='bold')
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 1].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center", 
                               color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        axes[0, 1].set_xticks(range(len(self.class_names)))
        axes[0, 1].set_yticks(range(len(self.class_names)))
        axes[0, 1].set_xticklabels(self.class_names, rotation=45)
        axes[0, 1].set_yticklabels(self.class_names)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Class Distribution
        unique, counts = np.unique(y_true, return_counts=True)
        colors = ['red', 'blue', 'green', 'orange']
        axes[1, 0].pie(counts, labels=[self.class_names[i] for i in unique], 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 0].set_title('Test Set Class Distribution', fontweight='bold')
        
        # Performance Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        x = np.arange(len(self.class_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, precision, width, label='Precision', alpha=0.8)
        axes[1, 1].bar(x, recall, width, label='Recall', alpha=0.8)
        axes[1, 1].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        axes[1, 1].set_title('Per-Class Performance Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.class_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.save_hash(filename, result_hash)
        print(f"💾 Saved: {filename}")
        plt.show()
    
    def save_detailed_report(self, y_true, y_pred, best_model_name, results, result_hash):
        filename = "7_detailed_results.txt"
        if not self.should_export(filename, result_hash):
            print(f"⏩ Skipping {filename} (already exists with same results)")
            return
            
        report_content = []
        report_content.append("="*60)
        report_content.append("BRAIN MRI CLASSIFICATION - SVM ANALYSIS RESULTS")
        report_content.append("="*60)
        report_content.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"\nBest Model: {best_model_name}")
        report_content.append(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        
        report_content.append("\nAll Model Results:")
        for name, result in results.items():
            report_content.append(f"  {name}: {result['accuracy']:.4f}")
        
        report_content.append("\nDetailed Classification Report:")
        class_report = classification_report(y_true, y_pred, target_names=self.class_names)
        report_content.append(class_report)
        
        report_content.append("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        report_content.append(str(cm))
        
        report_content.append("\nCustom Enhancements Implemented:")
        report_content.append("• Statistical features (mean, std, percentiles)")
        report_content.append("• Texture analysis using Local Binary Pattern")
        report_content.append("• Edge detection features")
        report_content.append("• Shape analysis using contours")
        report_content.append("• Multiple SVM kernel comparison")
        report_content.append("• Hyperparameter optimization")
        report_content.append("• Feature scaling and normalization")
        
        report_path = os.path.join(self.results_folder, filename)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        self.save_hash(filename, result_hash)
        print(f"💾 Saved: {filename}")

class ResultsAnalyzer:
    def __init__(self, class_names):
        self.class_names = class_names
        self.exporter = ResultsExporter(class_names)
    
    def plot_class_distribution(self, labels):
        plt.figure(figsize=(10, 5))
        
        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        
        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar([self.class_names[i] for i in unique], counts, 
                      color=['red', 'blue', 'green', 'orange'])
        plt.title('Class Distribution')
        plt.ylabel('Number of Images')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=[self.class_names[i] for i in unique], 
               autopct='%1.1f%%', colors=['red', 'blue', 'green', 'orange'])
        plt.title('Class Distribution (%)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_sample_images(self, images, labels, samples_per_class=3):
        fig, axes = plt.subplots(len(self.class_names), samples_per_class, 
                                figsize=(12, 10))
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get images for this class
            class_mask = labels == class_idx
            class_images = images[class_mask]
            
            # Select random samples
            if len(class_images) >= samples_per_class:
                sample_indices = np.random.choice(len(class_images), 
                                                samples_per_class, replace=False)
                
                for i, sample_idx in enumerate(sample_indices):
                    axes[class_idx, i].imshow(class_images[sample_idx], cmap='gray')
                    axes[class_idx, i].set_title(f'{class_name}')
                    axes[class_idx, i].axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results):
        plt.figure(figsize=(10, 6))
        
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('SVM Model Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def print_detailed_results(self, y_true, y_pred, best_model_name):
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION RESULTS")
        print("="*50)
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    
    def export_all_results(self, images, labels, results, y_true, y_pred, classifier, best_model_name):
        print("\nExporting results...")
        
        result_hash = self.exporter.generate_result_hash(results, y_pred)
        
        self.exporter.plot_class_distribution(labels, result_hash)
        self.exporter.plot_sample_images(images, labels, result_hash)
        self.exporter.plot_model_comparison(results, result_hash)
        self.exporter.plot_confusion_matrix(y_true, y_pred, best_model_name, result_hash)
        self.exporter.plot_feature_importance(classifier, result_hash)
        self.exporter.create_results_summary(results, y_true, y_pred, classifier, result_hash)
        self.exporter.save_detailed_report(y_true, y_pred, best_model_name, results, result_hash)
        
        print(f"\n Export complete! Check 'results/' folder")

def run_brain_mri_analysis(data_path):
    print("Brain MRI Classification with Custom SVM")
    print("="*50)
    
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    print("\n1. Loading Data...")
    loader = BrainMRILoader(data_path)
    images, labels = loader.load_images()

    print("\n2. Analyzing Dataset...")
    analyzer = ResultsAnalyzer(class_names)
    analyzer.plot_class_distribution(labels)
    print("Class distribution plotted")

    # Skip sample images for now to avoid hanging
    print("⏩ Skipping sample images visualization")


    print("\n3. Extracting Features...")
    classifier = EnhancedSVMClassifier()
    features = classifier.extract_features_from_images(images)

    print("\n4. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\n5. Training Models...")
    results, X_train_scaled, X_test_scaled = classifier.train_and_compare_models(
        X_train, y_train, X_test, y_test
    )
    
    print("\n6. Optimizing Best Model...")
    grid_search = classifier.optimize_best_model(X_train_scaled, y_train)
    
    print("\n7. Final Evaluation...")
    y_pred_final = classifier.best_model.predict(X_test_scaled)

    print("\n8. Visualizing Results...")
    analyzer.plot_model_comparison(results)
    analyzer.plot_confusion_matrix(y_test, y_pred_final)

    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    analyzer.print_detailed_results(y_test, y_pred_final, best_model_name)
    
    print("\n9. Exporting All Results...")
    analyzer.export_all_results(images, labels, results, y_test, y_pred_final, 
                               classifier, best_model_name)
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nDataset: {len(images)} images across 4 classes")
    print(f"Features: {features.shape[1]} custom features per image")
    print(f"Best Model: {best_model_name}")
    print(f"Final Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    
    
    return classifier, results, analyzer


if __name__ == "__main__":
    # Set your data path here
    data_path = "../dataset_11"  
    
    # Check if data exists
    if os.path.exists(data_path):
        # Run the analysis
        classifier, results, analyzer = run_brain_mri_analysis(data_path)
        print("\nAnalysis completed successfully!")
    else:
        print(f"Data folder '{data_path}' not found!")
        print("Please create the following folder structure:")
        print("data/")
        print("├── glioma/")
        print("├── meningioma/")
        print("├── notumor/")
        print("└── pituitary/")
        print("\nThen add your brain MRI images to each folder.")