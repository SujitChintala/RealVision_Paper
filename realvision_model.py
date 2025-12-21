"""
RealVision: CNN-based AI-Generated Image Detector
Transfer learning implementation using ResNet-18
"""

import torch
import torch.nn as nn
import torchvision.models as models


class RealVisionModel(nn.Module):
    """
    Transfer learning model for detecting AI-generated images.
    Uses ResNet-18 pre-trained on ImageNet with custom classification head.
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5, freeze_backbone=False):
        """
        Initialize RealVision model.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability for regularization
            freeze_backbone: If True, freeze ResNet-18 backbone weights
        """
        super(RealVisionModel, self).__init__()
        
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with custom classification head
        # Architecture: GAP (already in ResNet) -> FC -> BatchNorm -> Dropout -> Output
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(num_classes=2, dropout_rate=0.5, freeze_backbone=False):
    """
    Factory function to create RealVision model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        freeze_backbone: Whether to freeze backbone weights
    
    Returns:
        RealVisionModel instance
    """
    model = RealVisionModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print("RealVision Model Architecture:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
