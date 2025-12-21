"""
Testing and evaluation script for RealVision model
Generates comprehensive metrics and visualizations
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import json

from realvision_model import create_model


class RealVisionEvaluator:
    """Evaluator class for RealVision model."""
    
    def __init__(self, model_path, test_data_dir, device=None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            test_data_dir: Directory containing test data
            device: torch device (cuda/cpu)
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Load model and checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        
        # Create model
        self.model = create_model(
            num_classes=self.config.get('num_classes', 2),
            dropout_rate=self.config.get('dropout_rate', 0.5)
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Best validation accuracy: {self.checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        
        # Setup test data loader
        self.test_loader, self.class_names = self._setup_test_loader()
    
    def _setup_test_loader(self):
        """Setup test data loader."""
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(
            self.test_data_dir,
            transform=test_transform
        )
        
        print(f"Test samples: {len(test_dataset)}")
        print(f"Classes: {test_dataset.classes}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return test_loader, test_dataset.classes
    
    def evaluate(self):
        """Evaluate model on test set."""
        print("\nEvaluating model on test set...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Get confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Store results
        self.results = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {self.results['accuracy']*100:.2f}%")
        print(f"Precision: {self.results['precision']*100:.2f}%")
        print(f"Recall:    {self.results['recall']*100:.2f}%")
        print(f"F1-Score:  {self.results['f1_score']*100:.2f}%")
        print("=" * 50)
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            self.results['labels'],
            self.results['predictions'],
            target_names=self.class_names
        ))
    
    def plot_confusion_matrix(self, save_path='results'):
        """Plot and save confusion matrix."""
        os.makedirs(save_path, exist_ok=True)
        
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {filepath}")
        plt.close()
    
    def plot_training_history(self, save_path='results'):
        """Plot training history if available."""
        os.makedirs(save_path, exist_ok=True)
        
        if 'history' not in self.checkpoint:
            print("Training history not available in checkpoint.")
            return
        
        history = self.checkpoint['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(save_path, 'training_history.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {filepath}")
        plt.close()
    
    def save_results(self, save_path='results'):
        """Save evaluation results to JSON."""
        os.makedirs(save_path, exist_ok=True)
        
        results_dict = {
            'accuracy': float(self.results['accuracy']),
            'precision': float(self.results['precision']),
            'recall': float(self.results['recall']),
            'f1_score': float(self.results['f1_score']),
            'confusion_matrix': self.results['confusion_matrix'].tolist(),
            'class_names': self.class_names,
            'num_test_samples': len(self.results['labels'])
        }
        
        filepath = os.path.join(save_path, 'evaluation_results.json')
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Evaluation results saved to: {filepath}")
    
    def run_full_evaluation(self, save_path='results'):
        """Run complete evaluation pipeline."""
        # Evaluate
        self.evaluate()
        
        # Print results
        self.print_results()
        
        # Generate visualizations
        self.plot_confusion_matrix(save_path)
        self.plot_training_history(save_path)
        
        # Save results
        self.save_results(save_path)
        
        print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'checkpoints/best_model.pth'
    TEST_DATA_DIR = 'data/test'
    RESULTS_DIR = 'results'
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_realvision.py")
        exit(1)
    
    # Create evaluator
    evaluator = RealVisionEvaluator(
        model_path=MODEL_PATH,
        test_data_dir=TEST_DATA_DIR
    )
    
    # Run evaluation
    evaluator.run_full_evaluation(save_path=RESULTS_DIR)
