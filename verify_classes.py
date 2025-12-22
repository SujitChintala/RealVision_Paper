"""
Quick script to verify class indices
"""
from torchvision import datasets

# Check how ImageFolder loads the classes
train_dataset = datasets.ImageFolder('data/train')
print("Class to Index mapping:")
print(train_dataset.class_to_idx)
print("\nClasses in order:")
print(train_dataset.classes)
print(f"\nIndex 0 = {train_dataset.classes[0]}")
print(f"Index 1 = {train_dataset.classes[1]}")
