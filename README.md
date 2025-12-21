# RealVision: AI-Generated Image Detector

A transfer learning-based CNN system for detecting AI-generated images using ResNet-18 backbone.

## Features

- **Transfer Learning**: Pre-trained ResNet-18 on ImageNet
- **Custom Classification Head**: Global Average Pooling → FC → BatchNorm → Dropout → Output
- **Data Augmentation**: Random flips, rotations, color jitter
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Visualization**: Training curves and confusion matrix plots

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your data as follows:
```
data/
├── train/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

## Usage

### 1. Train the Model

```bash
python train_realvision.py
```

The script will:
- Automatically split training data into train (70%) and validation (30%)
- Apply data augmentation to training set
- Train for 20 epochs with learning rate scheduling
- Save the best model to `checkpoints/best_model.pth`
- Save training history to `checkpoints/training_history.json`

### 2. Evaluate the Model

```bash
python test_realvision.py
```

The script will:
- Load the best trained model
- Evaluate on test set
- Generate comprehensive metrics
- Create visualizations in `results/` folder
- Save evaluation results to JSON

### 3. Test Model Architecture

```bash
python realvision_model.py
```

## Configuration

You can modify training parameters in `train_realvision.py`:

```python
config = {
    'batch_size': 32,          # Batch size
    'epochs': 20,              # Number of epochs
    'learning_rate': 1e-4,     # Initial learning rate
    'dropout_rate': 0.5,       # Dropout probability
    'freeze_backbone': False,  # Freeze ResNet backbone
    'weight_decay': 1e-4,      # L2 regularization
}
```

## Model Architecture

- **Backbone**: ResNet-18 (pre-trained on ImageNet)
- **Input**: 224×224 RGB images
- **Classification Head**:
  - Linear layer (512 units)
  - Batch Normalization
  - ReLU activation
  - Dropout (0.5)
  - Linear layer (2 units for binary classification)

## Output Files

### Checkpoints (`checkpoints/`)
- `best_model.pth`: Best model based on validation accuracy
- `checkpoint_epoch_X.pth`: Periodic checkpoints
- `training_history.json`: Training and validation metrics

### Results (`results/`)
- `confusion_matrix.png`: Confusion matrix heatmap
- `training_history.png`: Training/validation loss and accuracy curves
- `evaluation_results.json`: Test set metrics

## Performance Metrics

The system reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Hardware Requirements

- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: Depends on dataset size

## Notes

- The model uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Training automatically uses GPU if available
- Data augmentation is only applied to training set
- Validation set is created by splitting 30% from training data

## Citation

```
RealVision: Detecting AI-Generated Images Using Convolutional Neural Networks
```
