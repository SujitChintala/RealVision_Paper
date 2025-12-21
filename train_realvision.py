"""
Training script for RealVision model
Implements transfer learning with data augmentation and validation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from realvision_model import create_model


class RealVisionTrainer:
    """Trainer class for RealVision model."""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup model
        self.model = create_model(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            freeze_backbone=config['freeze_backbone']
        ).to(self.device)
        
        # Setup loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Data augmentation and normalization for training
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Only normalization for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load full training dataset
        original_dataset = datasets.ImageFolder(
            self.config['train_data_dir'],
            transform=train_transform
        )
        
        # Save class information before any modifications
        class_names = original_dataset.classes
        
        # Limit dataset size if max_samples is specified
        if 'max_samples' in self.config and self.config['max_samples'] > 0:
            max_samples = min(self.config['max_samples'], len(original_dataset))
            indices = torch.randperm(len(original_dataset))[:max_samples]
            full_dataset = torch.utils.data.Subset(original_dataset, indices)
            print(f"Limited dataset to {max_samples} samples for faster training")
        else:
            full_dataset = original_dataset
        
        # Split into train and validation (70% train, 30% validation from train folder)
        train_size = int(0.7 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation dataset transform
        if isinstance(full_dataset, torch.utils.data.Subset):
            val_dataset.dataset.dataset.transform = val_transform
        else:
            val_dataset.dataset.transform = val_transform
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Classes: {class_names}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop."""
        print("\nStarting training...")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("-" * 50)
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best validation accuracy: {val_acc:.2f}%")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "=" * 50)
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best model saved at: {self.best_model_path}")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            filepath = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            self.best_model_path = filepath
            torch.save(checkpoint, filepath)
        else:
            filepath = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save(checkpoint, filepath)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config['checkpoint_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to: {history_path}")


def get_default_config():
    """Get default training configuration."""
    return {
        # Data
        'train_data_dir': 'data/train',
        'num_classes': 2,
        'max_samples': 2000,  # Limit total samples for faster training
        
        # Model
        'dropout_rate': 0.5,
        'freeze_backbone': False,
        
        # Training
        'batch_size': 64,  # Increased batch size for faster training
        'epochs': 10,  # Reduced from 20 for faster training
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        
        # Learning rate scheduler
        'lr_step_size': 7,
        'lr_gamma': 0.1,
        
        # System
        'num_workers': 4,
        'checkpoint_dir': 'checkpoints',
        'save_freq': 5
    }


if __name__ == "__main__":
    # Get configuration
    config = get_default_config()
    
    # Create trainer
    trainer = RealVisionTrainer(config)
    
    # Start training
    trainer.train()
